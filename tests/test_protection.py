"""版本化降额保护：起点/上限、早晚归因、来源冻结、回放一致性。"""

from __future__ import annotations

import math

import pytest

from cspplant.config import default_config
from cspplant.receiver.policy import HEADER, POLICY_FINGERPRINT
from cspplant.receiver.protection import (
    ENGINE_VERSION,
    ProtectionModel,
    calibration_ceiling_at,
    ceiling_at,
    effective_for_part,
    enter_threshold,
    projected_rate_c_s,
    threshold_fixed_point,
)
from cspplant.receiver.replay import (
    DERATE_CLEARED,
    DERATE_ENTERED,
    PARAMS_COMMITTED,
    interpret_decision,
    verify_protection_records,
)
from cspplant.receiver.self_params import CalibratedParams, ParamSet

LIMITS = default_config().receiver
CAL = CalibratedParams(8_000.0, 6.0, 1.0)
CP = 1.53


def test_projected_rate_and_enter_threshold_never_lag_calibration():
    params = ParamSet(
        capacity_kj_per_k=6_000.0,
        loss_kw_per_k=6.0,
        flow_gain=1.0,
        raw_capacity_kj_per_k=6_000.0,
        raw_loss_kw_per_k=6.0,
        raw_flow_gain=1.0,
        fit_ok=True,
        samples=300,
        quality=1.0,
    )
    rate = projected_rate_c_s(params, 120_000.0, 58.0, 600.0, 290.0, 22.0)
    cal_params = ParamSet(8_000.0, 6.0, 1.0, 8_000.0, 6.0, 1.0, False, 0, 0.0)
    cal_rate = projected_rate_c_s(cal_params, 120_000.0, 58.0, 600.0, 290.0, 22.0)
    assert rate > cal_rate  # 热容变小，外推升温更快
    threshold = enter_threshold(LIMITS, rate, lead_s=20.0)
    assert threshold < LIMITS.temp_derate_c  # 起点只能提前
    assert threshold >= LIMITS.temp_target_c
    # 零升温速率时起点与标定线一致，不得凭空提前。
    assert enter_threshold(LIMITS, 0.0, 20.0) == LIMITS.temp_derate_c


def test_self_ceiling_is_never_more_lenient_than_calibration():
    # 自算锚点提前（阈值更低）时，同温度下上限只能更低。
    self_threshold = LIMITS.temp_derate_c - 10.0
    temp = self_threshold + 20.0
    assert ceiling_at(temp, self_threshold, LIMITS) < calibration_ceiling_at(temp, LIMITS)


def test_derate_enters_early_and_explains_against_calibration_and_previous():
    model = ProtectionModel(LIMITS, CAL, inlet_c=290.0, ambient_c=22.0, derate_lead_s=2.0)
    # 直接把已确认的保守生效值放进状态：热容 6000。
    from dataclasses import replace

    tight = ParamSet(6_000.0, 6.0, 1.0, 6_000.0, 6.0, 1.0, True, 300, 1.0)
    model._state = replace(model.state, committed=tight, revision=1)
    model._filtered = tight
    # 在阈值不动点上判定：阈值温度处的速率反推出同一点作为起点。
    temp, rate_at_entry = threshold_fixed_point(
        tight, 60_000.0, 58.0, LIMITS, 290.0, 22.0, 2.0
    )
    decision = model.evaluate_with_context(
        temp + 5.0, 60_000.0, 58.0, 2.0, now=100.0,
        rate_c_s=projected_rate_c_s(tight, 60_000.0, 58.0, temp + 5.0, 290.0, 22.0),
    )
    assert decision.active is True
    assert LIMITS.temp_target_c < decision.enter_threshold_c < LIMITS.temp_derate_c
    assert decision.source == "self_pessimum"
    assert decision.ceiling_pct < 100.0
    explanation = decision.explanation
    assert explanation["temperature_margin_c"] > 0.0
    assert explanation["earlier_vs_calibration_s"] > 0.0
    assert explanation["revision"] == 1
    factor = next(f for f in explanation["factors"] if f["factor"] == "capacity_kj_per_k")
    assert factor["self"] < factor["calibration"]

    # 再来一次（先退出），对比上一次必须给出方向。
    clear_temp = decision.enter_threshold_c - 20.0
    clear_rate = projected_rate_c_s(tight, 60_000.0, 58.0, clear_temp, 290.0, 22.0)
    model.evaluate_with_context(
        clear_temp, 60_000.0, 58.0, 2.0, now=200.0, rate_c_s=clear_rate
    )
    decision2 = model.evaluate_with_context(
        temp, 60_000.0, 58.0, 2.0, now=300.0, rate_c_s=rate_at_entry
    )
    assert decision2.active is True
    assert "vs_previous" in decision2.explanation
    assert decision2.explanation["vs_previous"]["direction"] in ("earlier", "later", "same")
    assert model.entries() == 2


def test_effective_params_are_part_scoped():
    from dataclasses import replace

    tight = ParamSet(6_000.0, 9.0, 0.8, 6_000.0, 9.0, 0.8, True, 300, 1.0)
    state = replace(ProtectionModel(LIMITS, CAL).state, committed=tight)
    panel = effective_for_part(state, "panel")
    header = effective_for_part(state, HEADER)
    assert panel.capacity_kj_per_k == pytest.approx(6_000.0)
    assert panel.flow_gain == pytest.approx(0.8)
    # 集箱部位即使自算已收紧，保护也只按标定。
    assert header.capacity_kj_per_k == pytest.approx(8_000.0)
    assert header.flow_gain == pytest.approx(1.0)


def _entry_record(ts, decision, effective, lead_s=20.0):
    return {
        "ts": ts,
        "action": DERATE_ENTERED,
        "payload": {
            "decision": decision,
            "effective": effective.as_dict(),
            "derate_lead_s": lead_s,
        },
    }


def test_replay_interprets_with_recorded_version_and_matches():
    model = ProtectionModel(LIMITS, CAL, inlet_c=290.0, ambient_c=22.0)
    from dataclasses import replace

    tight = ParamSet(6_000.0, 6.0, 1.0, 6_000.0, 6.0, 1.0, True, 300, 1.0)
    model._state = replace(model.state, committed=tight, revision=2)
    model._filtered = tight
    threshold, rate = threshold_fixed_point(
        tight, 120_000.0, 58.0, LIMITS, 290.0, 22.0, 20.0
    )
    decision = model.evaluate_with_context(threshold, 120_000.0, 58.0, 2.0, now=10.0, rate_c_s=rate)
    record = _entry_record(10.0, decision.as_dict(), tight)
    verdict = verify_protection_records([record], LIMITS)
    assert verdict.ok is True
    assert verdict.checked == 1

    # 直接解释：旧引擎版本必须被拒（不能用新引擎冒充）。
    old = dict(decision.as_dict())
    old["engine_version"] = ENGINE_VERSION + 1
    record_old = _entry_record(10.0, old, tight)
    verdict_old = verify_protection_records([record_old], LIMITS)
    assert verdict_old.ok is False
    assert any(item["issue"] == "unknown_engine" for item in verdict_old.issues)


def test_replay_flags_tampered_ceiling_and_policy_drift():
    model = ProtectionModel(LIMITS, CAL, inlet_c=290.0, ambient_c=22.0)
    from dataclasses import replace

    tight = ParamSet(6_000.0, 6.0, 1.0, 6_000.0, 6.0, 1.0, True, 300, 1.0)
    model._state = replace(model.state, committed=tight, revision=1)
    model._filtered = tight
    threshold, rate = threshold_fixed_point(
        tight, 120_000.0, 58.0, LIMITS, 290.0, 22.0, 20.0
    )
    decision = model.evaluate_with_context(threshold, 120_000.0, 58.0, 2.0, now=10.0, rate_c_s=rate)
    tampered = dict(decision.as_dict())
    tampered["ceiling_pct"] = 99.0  # 被人往宽松方向改
    verdict = verify_protection_records([_entry_record(10.0, tampered, tight)], LIMITS)
    assert verdict.ok is False
    assert any(
        item["issue"] == "replay_mismatch" and item["field"] == "ceiling_pct"
        for item in verdict.issues
    )

    drifted = dict(decision.as_dict())
    drifted["policy_fingerprint"] = "deadbeef0000"
    verdict_drift = verify_protection_records([_entry_record(10.0, drifted, tight)], LIMITS)
    assert any(item["issue"] == "policy_drift" for item in verdict_drift.issues)


def test_revision_chain_must_be_monotonic():
    good = [
        {"ts": 1.0, "action": PARAMS_COMMITTED, "payload": {"revision": 1}},
        {"ts": 2.0, "action": PARAMS_COMMITTED, "payload": {"revision": 2}},
    ]
    assert verify_protection_records(good, LIMITS).ok is True
    bad = [
        {"ts": 1.0, "action": PARAMS_COMMITTED, "payload": {"revision": 3}},
        {"ts": 2.0, "action": PARAMS_COMMITTED, "payload": {"revision": 1}},
    ]
    verdict = verify_protection_records(bad, LIMITS)
    assert any(item["issue"] == "revision_chain_broken" for item in verdict.issues)


def test_decision_carries_all_version_stamps():
    model = ProtectionModel(LIMITS, CAL, inlet_c=290.0, ambient_c=22.0)
    decision = model.evaluate(600.0, 50_000.0, 58.0, 2.0)
    assert decision.engine_version == ENGINE_VERSION
    assert decision.policy_fingerprint == POLICY_FINGERPRINT
    assert decision.schema == 2
    assert decision.part == "panel"
    assert decision.revision == 0
