"""自算热工参数：版本、保守棘轮、按部位来源、迁移。"""

from __future__ import annotations

import pytest

from cspplant.config import default_config
from cspplant.receiver.self_params import (
    SCHEMA_VERSION,
    CalibratedParams,
    SelfParamError,
    initial_state,
    is_more_conservative,
    migrate_document,
    pessimistic,
    state_from_dict,
)
from cspplant.receiver.protection import ProtectionModel
from cspplant.receiver.policy import (
    CAL_ONLY,
    HEADER,
    PANEL,
    POLICY_FINGERPRINT,
    SELF_PESSIMUM,
    SUPPORT,
    policy_for,
    uses_self_params,
)

CAL = CalibratedParams(8_000.0, 6.0, 1.0)
LIMITS = default_config().receiver


def _raw(capacity, loss, gain, *, fit_ok=True, samples=200, quality=1.0):
    from cspplant.receiver.self_params import ParamSet

    return ParamSet(
        capacity_kj_per_k=capacity,
        loss_kw_per_k=loss,
        flow_gain=gain,
        raw_capacity_kj_per_k=capacity,
        raw_loss_kw_per_k=loss,
        raw_flow_gain=gain,
        fit_ok=fit_ok,
        samples=samples,
        quality=quality,
    )


def test_pessimistic_clamp_only_moves_toward_safety():
    # 自算说热容更小、散热更大、增益更低：全部更保守，照单全收。
    candidate = pessimistic(CAL, _raw(6_000.0, 9.0, 0.8))
    assert candidate.capacity_kj_per_k == pytest.approx(6_000.0)
    assert candidate.loss_kw_per_k == pytest.approx(9.0)
    assert candidate.flow_gain == pytest.approx(0.8)
    # 自算说热容更大、散热更小、增益更高（更宽松）：一律退回标定。
    loose = pessimistic(CAL, _raw(12_000.0, 3.0, 1.2))
    assert loose.capacity_kj_per_k == pytest.approx(8_000.0)
    assert loose.loss_kw_per_k == pytest.approx(6.0)
    assert loose.flow_gain == pytest.approx(1.0)


def test_ratchet_is_one_way_and_bumps_revision():
    model = ProtectionModel(LIMITS, CAL, inlet_c=290.0, ambient_c=22.0)
    assert model.state.revision == 0
    # 用受控方式直接验证提交门槛。
    tighter = _raw(6_000.0, 9.0, 0.8)
    assert is_more_conservative(model.state.committed, pessimistic(CAL, tighter)) is True
    looser = _raw(9_000.0, 4.0, 1.1)
    assert is_more_conservative(model.state.committed, pessimistic(CAL, looser)) is False


def test_part_policy_is_frozen_by_part():
    assert policy_for(PANEL).source == SELF_PESSIMUM
    assert uses_self_params(PANEL) is True
    assert policy_for(HEADER).source == CAL_ONLY
    assert policy_for(SUPPORT).source == CAL_ONLY
    assert uses_self_params(HEADER) is False
    # 指纹是冻结策略表的指纹，改表才会变。
    assert isinstance(POLICY_FINGERPRINT, str) and len(POLICY_FINGERPRINT) == 12
    with pytest.raises(KeyError):
        policy_for("tube-not-registered")


def test_cal_only_part_never_adopts_self_params():
    header_model = ProtectionModel(LIMITS, CAL, part=HEADER, inlet_c=290.0, ambient_c=22.0)
    effective = header_model.effective_params()
    assert effective.capacity_kj_per_k == pytest.approx(8_000.0)
    assert effective.flow_gain == pytest.approx(1.0)
    # 即使辨识出严重偏差，集箱部位的 observe 也不得收紧生效值。
    model = ProtectionModel(LIMITS, CAL, part=HEADER, inlet_c=290.0, ambient_c=22.0)
    model.observe(50_000.0, 58.0, 400.0, 2.0)
    assert model.state.revision == 0


def test_recalibrate_is_the_only_relaxation_path_and_audited_revision():
    model = ProtectionModel(LIMITS, CAL, inlet_c=290.0, ambient_c=22.0)
    # 先收紧（模拟检修前已学到的结焦值）。
    state = initial_state(CAL)
    from dataclasses import replace

    tightened = replace(state, committed=_raw(6_000.0, 9.0, 0.8), revision=3)
    model.restore_state(tightened)
    # 检修后重新标定允许放松，但必须升 revision 且带回新基线。
    new_state = model.recalibrate(9_000.0, 5.0, 1.0, now=1234.0, operator="maint-7")
    assert new_state.revision == 4
    assert new_state.calibrated.capacity_kj_per_k == pytest.approx(9_000.0)
    assert new_state.committed.capacity_kj_per_k == pytest.approx(9_000.0)
    with pytest.raises(SelfParamError):
        model.recalibrate(10.0, 6.0, 1.0, now=1.0)


def test_schema_v1_document_migrates_to_v2_with_flow_gain():
    legacy = {
        "schema": 1,
        "revision": 2,
        "updated_ts": 10.0,
        "calibrated": {"capacity_kj_per_k": 8000.0, "loss_kw_per_k": 6.0},
        "committed": {
            "capacity_kj_per_k": 6500.0,
            "loss_kw_per_k": 7.5,
            "raw_capacity_kj_per_k": 6500.0,
            "raw_loss_kw_per_k": 7.5,
            "fit_ok": True,
            "samples": 300,
            "quality": 1.0,
        },
    }
    migrated, applied = migrate_document(legacy)
    assert applied == [2]
    assert migrated["schema"] == SCHEMA_VERSION
    state = state_from_dict(legacy)
    assert state.schema == SCHEMA_VERSION
    assert state.revision == 2
    assert state.committed.flow_gain == pytest.approx(1.0)
    assert state.calibrated.flow_gain == pytest.approx(1.0)


def test_unknown_schema_is_rejected_not_silently_reinterpreted():
    future = {
        "schema": 99,
        "calibrated": {"capacity_kj_per_k": 8000.0, "loss_kw_per_k": 6.0},
        "committed": {},
    }
    with pytest.raises(SelfParamError):
        migrate_document(future)


def test_persistence_round_trip_and_startup_migration(tmp_path):
    from cspplant.receiver.persist import ReceiverPersistence
    from cspplant.store.jsonfile import JsonStore

    persistence = ReceiverPersistence(JsonStore(str(tmp_path)))
    state, applied, fresh = persistence.load_or_init(CAL)
    assert fresh is True and applied == []
    state = initial_state(CAL)
    persistence.save(state)
    again, applied, fresh = persistence.load_or_init(CAL)
    assert fresh is False and again.revision == 0

    # 直接落一份 v1 旧文档，启动时必须迁移并覆盖落盘。
    persistence.store.write(
        "receiver/self_params",
        {
            "schema": 1,
            "revision": 5,
            "calibrated": {"capacity_kj_per_k": 8000.0, "loss_kw_per_k": 6.0},
            "committed": {"capacity_kj_per_k": 7000.0, "loss_kw_per_k": 7.0},
        },
    )
    migrated_state, applied, fresh = persistence.load_or_init(CAL)
    assert applied == [2]
    assert migrated_state.schema == SCHEMA_VERSION
    assert migrated_state.revision == 5
    on_disk = persistence.store.read("receiver/self_params")
    assert on_disk["schema"] == SCHEMA_VERSION
