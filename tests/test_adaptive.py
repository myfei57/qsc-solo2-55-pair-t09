"""吸热器自算热工参数：估计、迁移、部位裁决、单调保守与回放一致性。"""

from __future__ import annotations

import dataclasses

import pytest

from cspplant.app import build_plant
from cspplant.audit.record import AuditRecord
from cspplant.clock import ManualClock
from cspplant.config import AdaptiveLimits, default_config
from cspplant.console.api import Api
from cspplant.errors import ConfigError
from cspplant.receiver.adaptive import (
    CALIBRATION,
    LEARNED,
    AdaptiveProtection,
    ParameterStore,
    ThermalEstimate,
    ThermalEstimator,
    replay_verdicts,
)
from cspplant.receiver.derate import DerateController
from cspplant.receiver.flow import FlowGate
from cspplant.receiver.header import SaltHeader
from cspplant.receiver.receiver import Receiver
from cspplant.receiver.sensor import TemperatureSensor
from cspplant.receiver.thermal import (
    INLET_TEMP_C,
    LOSS_KW_K,
    RECEIVER_CAPACITY_KJ_K,
    ThermalModel,
)
from cspplant.receiver.threshold import ThresholdModel
from cspplant.receiver.valve import InletValve


def make_config(**adaptive_overrides):
    """默认配置 + 小窗口自算参数，便于测试快速收敛。"""

    config = default_config()
    overrides = {"window": 24, "min_temp_rise_c": 2.0, "min_power_swing_kw": 500.0}
    overrides.update(adaptive_overrides)
    return dataclasses.replace(
        config, adaptive=dataclasses.replace(config.adaptive, **overrides)
    )


def make_store(config=None) -> ParameterStore:
    config = config or make_config()
    return ParameterStore(
        config.receiver, config.adaptive, reference_flow_kg_s=config.salt.flow_baseline_kg_s
    )


def estimate(capacity: float, loss: float, ts: float = 1.0) -> ThermalEstimate:
    return ThermalEstimate(
        capacity_kj_k=capacity,
        loss_kw_k=loss,
        samples=30,
        temp_span_c=12.0,
        power_swing_kw=6000.0,
        residual_ratio=0.01,
        created_ts=ts,
    )


def step(protection: AdaptiveProtection, temp_now: float, now: float, temp_prev: float | None = None):
    """喂一个受热步并按当前温度判定（量测与判定同温）。"""

    protection.feed(
        power_kw=20_000.0,
        flow_kg_s=58.0,
        temp_prev_c=temp_now if temp_prev is None else temp_prev,
        temp_now_c=temp_now,
        dt=1.0,
        now=now,
    )
    return protection.judge(temp_now)


def build_adaptive_receiver(config) -> Receiver:
    store = make_store(config)
    protection = AdaptiveProtection(
        config.receiver,
        config.adaptive,
        store,
        ThermalEstimator(config.receiver, config.adaptive, INLET_TEMP_C),
    )
    return Receiver(
        config.receiver,
        InletValve(config.receiver),
        FlowGate(config.receiver),
        TemperatureSensor(config.receiver.ambient_c),
        ThermalModel(config.receiver),
        DerateController(config.receiver),
        SaltHeader(config.salt),
        ThresholdModel(config.salt),
        adaptive=protection,
    )


# ---- 估计器 -------------------------------------------------------------


def test_estimator_recovers_thermal_constants():
    config = make_config()
    estimator = ThermalEstimator(config.receiver, config.adaptive, INLET_TEMP_C)
    model = ThermalModel(config.receiver)
    temp = model.value_c()
    for index in range(300):
        power = 20_000.0 + (12_000.0 if (index // 8) % 2 else 0.0)
        nxt = model.update(power, 58.0, 1.0)
        estimator.feed(power, 58.0, temp, nxt, 1.0)
        temp = nxt
    result = estimator.estimate(300.0)
    assert result is not None
    assert result.capacity_kj_k == pytest.approx(RECEIVER_CAPACITY_KJ_K, rel=0.05)
    assert result.loss_kw_k == pytest.approx(LOSS_KW_K, rel=0.1)
    assert result.samples == config.adaptive.window


def test_estimator_stays_silent_without_excitation():
    config = make_config()
    estimator = ThermalEstimator(config.receiver, config.adaptive, INLET_TEMP_C)
    model = ThermalModel(config.receiver)
    temp = model.value_c()
    for _ in range(2_000):
        temp = model.update(25_000.0, 58.0, 1.0)
    for _ in range(100):
        nxt = model.update(25_000.0, 58.0, 1.0)
        estimator.feed(25_000.0, 58.0, temp, nxt, 1.0)
        temp = nxt
    assert estimator.ready() is True
    assert estimator.estimate(100.0) is None


def test_estimator_ignores_low_flow_samples():
    config = make_config()
    estimator = ThermalEstimator(config.receiver, config.adaptive, INLET_TEMP_C)
    for index in range(50):
        estimator.feed(20_000.0, 1.0, 300.0 + index, 301.0 + index, 1.0)
    assert estimator.samples() == 0


def test_estimator_stays_silent_across_regime_change():
    """窗口横跨结焦突变时残差超限，失真估计不得流出。"""

    config = make_config()
    estimator = ThermalEstimator(config.receiver, config.adaptive, INLET_TEMP_C)
    model = ThermalModel(config.receiver)
    temp = model.value_c()

    def drive(steps, offset):
        nonlocal temp
        for index in range(steps):
            power = 20_000.0 + (12_000.0 if ((index + offset) // 8) % 2 else 0.0)
            nxt = model.update(power, 58.0, 1.0)
            estimator.feed(power, 58.0, temp, nxt, 1.0)
            temp = nxt

    drive(300, 0)
    assert estimator.estimate(300.0) is not None
    model.apply_fouling(0.5, 0.6)
    drive(12, 3)
    # 窗口内新旧真值各占一半，单一参数组拟合必然失真。
    assert estimator.estimate(312.0) is None
    drive(12, 7)
    recovered = estimator.estimate(324.0)
    assert recovered is not None
    assert recovered.capacity_kj_k == pytest.approx(RECEIVER_CAPACITY_KJ_K * 0.5, rel=0.1)
    assert recovered.loss_kw_k == pytest.approx(LOSS_KW_K * 0.6, rel=0.1)


# ---- 版本库：单调保守 ----------------------------------------------------


def test_calibration_version_is_v0():
    store = make_store()
    current = store.current()
    assert current.version == 0
    assert current.source == CALIBRATION
    assert current.derate_start_c == pytest.approx(620.0)
    assert current.ceiling_cap_pct == pytest.approx(100.0)


def test_propose_accepts_only_conservative_direction():
    store = make_store()
    result = store.propose(estimate(7000.0, 5.0), 1.0)
    assert result.migrated is True
    assert store.current().version == 1
    assert store.current().capacity_kj_k == pytest.approx(7000.0)
    assert store.current().derate_start_c == pytest.approx(600.0 + 20.0 * 0.875)

    optimistic = store.propose(estimate(9000.0, 8.0), 2.0)
    assert optimistic.migrated is False
    assert store.current().version == 1
    assert store.current().capacity_kj_k == pytest.approx(7000.0)
    assert store.current().loss_kw_k == pytest.approx(5.0)
    assert store.rejections() == 1


def test_propose_clamps_to_floor():
    store = make_store()
    result = store.propose(estimate(500.0, 0.1), 1.0)
    assert result.migrated is True
    assert result.clamped is True
    current = store.current()
    assert current.capacity_kj_k == pytest.approx(RECEIVER_CAPACITY_KJ_K * 0.5)
    assert current.loss_kw_k == pytest.approx(LOSS_KW_K * 0.4)


def test_margin_blocks_noise_driven_migration():
    store = make_store()
    result = store.propose(estimate(7950.0, 5.99), 1.0)
    assert result.migrated is False
    assert result.reason == "gain_below_margin"
    assert store.current().version == 0


def test_versions_persist_across_reload(tmp_path):
    from cspplant.store.jsonfile import JsonStore

    config = make_config()
    root = tmp_path / "state"
    store = ParameterStore(
        config.receiver,
        config.adaptive,
        reference_flow_kg_s=58.0,
        persistence=JsonStore(str(root)),
    )
    store.propose(estimate(6500.0, 4.5), 1.0)
    store.propose(estimate(6000.0, 4.2), 2.0)

    reloaded = ParameterStore(
        config.receiver,
        config.adaptive,
        reference_flow_kg_s=58.0,
        persistence=JsonStore(str(root)),
    )
    assert reloaded.current().version == 2
    assert reloaded.current().capacity_kj_k == pytest.approx(6000.0)
    assert reloaded.get(1).loss_kw_k == pytest.approx(4.5)
    assert len(reloaded.versions()) == 3


# ---- 部位裁决矩阵 --------------------------------------------------------


def test_trip_zone_is_pinned_to_calibration():
    base = {
        "enabled": True,
        "window": 24,
        "min_temp_rise_c": 2.0,
        "min_power_swing_kw": 500.0,
        "capacity_floor_ratio": 0.5,
        "loss_floor_ratio": 0.4,
        "migration_margin_pct": 1.0,
    }
    with pytest.raises(ConfigError):
        AdaptiveLimits.from_mapping(
            {**base, "zone_sources": {"panel": "learned", "trip": "learned"}}
        )
    with pytest.raises(ConfigError):
        AdaptiveLimits.from_mapping({**base, "zone_sources": {"panel": "learned"}})
    with pytest.raises(ConfigError):
        AdaptiveLimits.from_mapping(
            {**base, "zone_sources": {"panel": "maybe", "trip": "calibration"}}
        )


def test_panel_on_calibration_ignores_learned_versions():
    config = make_config(zone_sources={"panel": "calibration", "trip": "calibration"})
    store = make_store(config)
    protection = AdaptiveProtection(config.receiver, config.adaptive, store)
    store.propose(estimate(5000.0, 4.0), 1.0)
    assert store.current().version == 1
    verdict = step(protection, 619.0, 2.0, temp_prev=610.0)
    assert verdict.source == CALIBRATION
    assert verdict.version == 0
    assert verdict.derate_start_c == pytest.approx(config.receiver.temp_derate_c)


def test_learned_derate_starts_earlier_and_explains():
    config = make_config()
    store = make_store(config)
    protection = AdaptiveProtection(config.receiver, config.adaptive, store)
    step(protection, 500.0, 1.0, temp_prev=499.0)
    store.propose(estimate(4800.0, 4.2), 2.0)
    verdict = step(protection, 501.0, 3.0, temp_prev=500.0)
    assert verdict.version == 1
    assert verdict.derate_start_c == pytest.approx(600.0 + 20.0 * 0.6)
    assert "v0→v1" in verdict.explanation
    assert "提前" in verdict.explanation

    below = step(protection, 611.0, 4.0, temp_prev=610.0)
    assert below.active is False
    crossed = step(protection, 612.5, 5.0, temp_prev=611.0)
    assert crossed.active is True
    assert "进入降额" in crossed.explanation


# ---- 回放一致性 ----------------------------------------------------------


def _verdict_record(store: ParameterStore, version_no: int, temp_c: float, ts: float) -> AuditRecord:
    version = store.get(version_no)
    ceiling = min(100.0, version.ceiling_cap_pct)
    if temp_c > version.derate_start_c:
        span = store.limits.temp_max_c - version.derate_start_c
        ceiling = min(100.0 * (store.limits.temp_max_c - temp_c) / span, version.ceiling_cap_pct)
    return AuditRecord.create(
        ts,
        "",
        "receiver",
        "adaptive_verdict",
        "notice",
        "",
        version=version_no,
        derate_start_c=round(version.derate_start_c, 3),
        ceiling_cap_pct=round(version.ceiling_cap_pct, 3),
        temperature_c=temp_c,
        ceiling_pct=round(ceiling, 3),
    )


def test_replay_confirms_consistent_records():
    store = make_store()
    store.propose(estimate(6000.0, 5.0), 1.0)
    records = [
        _verdict_record(store, 0, 500.0, 1.0),
        _verdict_record(store, 1, 500.0, 2.0),
        _verdict_record(store, 1, 615.0, 3.0),
    ]
    report = replay_verdicts(store, records)
    assert report["checked"] == 3
    assert report["consistent"] is True
    assert report["mismatches"] == []


def test_replay_detects_tampered_records():
    store = make_store()
    store.propose(estimate(6000.0, 5.0), 1.0)
    good = _verdict_record(store, 1, 500.0, 1.0)
    tampered = AuditRecord.create(
        2.0,
        "",
        "receiver",
        "adaptive_verdict",
        "notice",
        "",
        version=1,
        derate_start_c=round(store.current().derate_start_c + 5.0, 3),
        ceiling_cap_pct=round(store.current().ceiling_cap_pct, 3),
        temperature_c=500.0,
        ceiling_pct=round(min(100.0, store.current().ceiling_cap_pct), 3),
    )
    missing = AuditRecord.create(
        3.0, "", "receiver", "adaptive_verdict", "notice", "", version=99
    )
    report = replay_verdicts(store, [good, tampered, missing])
    assert report["checked"] == 2
    assert report["consistent"] is False
    reasons = {item["reason"] for item in report["mismatches"]}
    assert reasons == {"recompute_diverged", "version_not_in_store"}
    diverged = [item for item in report["mismatches"] if item["reason"] == "recompute_diverged"]
    assert diverged[0]["fields"] == ["derate_start_c"]


# ---- 吸热器集成 ----------------------------------------------------------


def test_receiver_without_adaptive_keeps_fixed_behavior():
    config = default_config()
    receiver = Receiver(
        config.receiver,
        InletValve(config.receiver),
        FlowGate(config.receiver),
        TemperatureSensor(config.receiver.ambient_c),
        ThermalModel(config.receiver),
        DerateController(config.receiver),
        SaltHeader(config.salt),
        ThresholdModel(config.salt),
    )
    assert receiver.adaptive is None
    assert receiver.last_adaptive_verdict() is None
    mid = (config.receiver.temp_derate_c + config.receiver.temp_max_c) / 2.0
    assert receiver.derate.ceiling_pct(mid) == pytest.approx(50.0)


def test_receiver_with_adaptive_reports_verdict_and_zeroes_on_low_flow():
    config = make_config()
    receiver = build_adaptive_receiver(config)
    limits = receiver.limits
    receiver.verify_flow(limits.flow_min_kg_s + 16.0, limits.flow_confirm_s)
    receiver.admit()
    report = receiver.heat(25_000.0, 58.0, 1.0, now=1.0)
    assert "adaptive" in report
    assert report["adaptive"]["version"] == 0
    assert receiver.last_adaptive_verdict() is not None
    assert receiver.ceiling_pct(20.0) == pytest.approx(0.0)


# ---- 平台级：结焦 → 迁移 → 审计 → 回放 ------------------------------------


def _bring_online(platform) -> None:
    platform.start()
    platform.track(500.0, 720.0)
    platform.circulate()
    platform.dispatch(100.0, "test")


def test_fouling_drives_conservative_migration_and_consistent_replay(tmp_path):
    config = make_config()
    platform = build_plant(config, clock=ManualClock(), storage_root=str(tmp_path / "state"))
    _bring_online(platform)
    for _ in range(120):
        platform.cycle(dt=1.0)
    v0 = platform.adaptive_params.current()
    assert v0.version == 0
    assert platform.adaptive_params.rejections() >= 1

    platform.receiver.thermal.apply_fouling(0.6, 0.7)
    for _ in range(240):
        platform.cycle(dt=1.0)

    current = platform.adaptive_params.current()
    assert current.version >= 1
    assert current.source == LEARNED
    assert current.capacity_kj_k < v0.capacity_kj_k
    assert current.loss_kw_k < v0.loss_kw_k
    assert current.derate_start_c < config.receiver.temp_derate_c
    assert current.ceiling_cap_pct < 100.0
    assert platform.audit.count("adaptive_migration") >= 1
    assert platform.audit.count("adaptive_verdict") >= 1

    verdict = platform.receiver.last_adaptive_verdict()
    assert verdict is not None
    assert verdict.version == current.version

    report = platform.verify_adaptive_replay()
    assert report["checked"] >= 1
    assert report["consistent"] is True
    assert platform.audit.count("adaptive_replay") == 1

    status = platform.status()["adaptive"]
    assert status["current"]["version"] == current.version
    assert status["zones"]["trip"]["source"] == "calibration"


def test_versions_survive_platform_restart(tmp_path):
    config = make_config()
    root = str(tmp_path / "state")
    platform = build_plant(config, clock=ManualClock(), storage_root=root)
    _bring_online(platform)
    platform.receiver.thermal.apply_fouling(0.6, 0.7)
    for _ in range(240):
        platform.cycle(dt=1.0)
    current = platform.adaptive_params.current()
    assert current.version >= 1

    rebuilt = build_plant(config, clock=ManualClock(), storage_root=root)
    restored = rebuilt.adaptive_params.current()
    assert restored.version == current.version
    assert restored.derate_start_c == pytest.approx(current.derate_start_c)
    assert rebuilt.adaptive.status()["zone_sources"]["panel"] == "learned"


def test_console_exposes_adaptive_report_and_replay(tmp_path):
    config = make_config()
    platform = build_plant(config, clock=ManualClock(), storage_root=str(tmp_path / "state"))
    _bring_online(platform)
    platform.receiver.thermal.apply_fouling(0.6, 0.7)
    for _ in range(240):
        platform.cycle(dt=1.0)

    api = Api(platform)
    status, payload = api.handle("GET", "/api/adaptive", {}, {})
    assert status == 200
    assert payload["current"]["version"] >= 1
    assert payload["zone_sources"] == {"panel": "learned", "trip": "calibration"}
    assert payload["last_verdict"]["version"] == payload["current"]["version"]
    assert len(payload["versions"]) == payload["current"]["version"] + 1

    status, payload = api.handle("POST", "/api/adaptive/replay", {}, {})
    assert status == 200
    assert payload["consistent"] is True
    assert payload["checked"] >= 1


def test_adaptive_disabled_keeps_receiver_on_fixed_thresholds(tmp_path):
    config = make_config(enabled=False)
    platform = build_plant(config, clock=ManualClock(), storage_root=str(tmp_path / "state"))
    _bring_online(platform)
    for _ in range(60):
        platform.cycle(dt=1.0)
    assert platform.receiver.adaptive is None
    assert platform.receiver.last_adaptive_verdict() is None
    assert platform.adaptive_params.current().version == 0
    assert platform.status()["adaptive"]["enabled"] is False
