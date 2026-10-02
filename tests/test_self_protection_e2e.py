"""平台级自算保护闭环：漂移学习、审计落盘、回放核对、跨重启版本保留。"""

from __future__ import annotations

import math

import pytest

from cspplant.app import build_plant
from cspplant.clock import ManualClock
from cspplant.config import default_config

# 端到端用例需要持续日照，时钟从正午起步。
NOON = 12 * 3600.0


def _noon_platform(tmp_path):
    return build_plant(
        default_config(), clock=ManualClock(NOON), storage_root=str(tmp_path / "state")
    )


def _drive_blocks(platform, dni_values, *, cycles=15, dt=2.0):
    for dni in dni_values:
        platform.track(float(dni), (platform.clock.now() / 60.0) % 1440.0)
        entered = False
        for _ in range(cycles):
            report = platform.cycle(dt=dt)
            if report["heat"].get("derate", {}).get("changed") == "entered":
                entered = True
                break
        if entered:
            return True
    return False


def test_self_params_protects_earlier_after_coking_and_replay_matches(tmp_path):
    platform = _noon_platform(tmp_path)
    platform.start()
    platform.track(950.0, 720.0)
    platform.circulate()
    platform.dispatch(100.0, "e2e")
    for _ in range(30):
        platform.cycle(dt=2.0)

    # 标定参数下先做一段功率扫掠，让辨识器建立可信基线。
    baseline_blocks = [700, 850, 1000, 850, 700, 900, 1050, 900, 750, 950]
    _drive_blocks(platform, baseline_blocks)

    # 注入结焦：热容下降、流量计漂低（读数高估）。
    drift = platform.degrade_receiver(capacity_kj_per_k=6_500.0, flow_gain=0.85)
    assert drift["capacity_kj_per_k"] == 6_500.0

    # 高负荷变工况：自算应收紧并在标定线之前进入降额。
    high_blocks = [800, 950, 1100, 950, 800, 1000, 1100, 1050, 1000, 1050, 1100, 1080]
    entered = _drive_blocks(platform, high_blocks)
    assert entered is True

    status = platform.status()
    self_params = status["self_params"]
    assert self_params["schema"] == 2
    assert self_params["revision"] >= 1
    # 生效热容只能比标定更保守（更小）。
    assert self_params["committed"]["capacity_kj_per_k"] <= 8_000.0
    # 按部位来源定死：管屏允许自算接管。
    assert self_params["source_policy"] == "self_pessimum"

    # 审计里必须有版本化的进入降额记录。
    entries = [
        item
        for item in platform.audit.entries(limit=200)
        if item["action"] == "derate_entered"
    ]
    assert entries, "进入降额必须落审计"
    decision = entries[0]["payload"]["decision"]
    assert decision["engine_version"] == 1
    assert decision["enter_threshold_c"] < platform.config.receiver.temp_derate_c
    assert decision["explanation"]["earlier_vs_calibration_s"] > 0.0
    assert "capacity_kj_per_k" in [f["factor"] for f in decision["explanation"]["factors"]]

    # 回放按当时版本解释，必须与落盘一致。
    verdict = platform.verify_protection_replay()
    assert verdict["ok"] is True
    assert verdict["checked"] >= 1
    assert verdict["issues"] == []


def test_self_params_survive_restart_with_revision(tmp_path):
    clock = ManualClock(NOON)
    platform = build_plant(
        default_config(), clock=clock, storage_root=str(tmp_path / "state")
    )
    platform.start()
    platform.track(950.0, 720.0)
    platform.circulate()
    platform.dispatch(100.0, "restart")
    for _ in range(30):
        platform.cycle(dt=2.0)
    _drive_blocks(platform, [700, 850, 1000, 850, 700, 900, 1050, 900, 750, 950])
    platform.degrade_receiver(capacity_kj_per_k=6_500.0, flow_gain=0.85)
    _drive_blocks(platform, [800, 950, 1100, 950, 800, 1000, 1100, 1050, 1000, 1050, 1100])
    revision_before = platform.receiver.protection.state.revision
    capacity_before = platform.receiver.protection.state.committed.capacity_kj_per_k

    restarted = build_plant(
        default_config(), clock=clock, storage_root=platform.store.root
    )
    # 棘轮值与 revision 跨重启原样带回，不得悄悄回到出厂标定。
    assert restarted.receiver.protection.state.revision == revision_before
    assert restarted.receiver.protection.state.committed.capacity_kj_per_k == pytest.approx(
        capacity_before, abs=1e-3
    )
    # 重启后的平台用同一版本解释历史，回放仍一致。
    verdict = restarted.verify_protection_replay()
    assert verdict["ok"] is True


def test_recalibrate_after_maintenance_relaxes_with_operator_audit(platform):
    platform.start()
    protection = platform.receiver.protection
    # 模拟已收紧到 6000。
    from dataclasses import replace

    from cspplant.receiver.self_params import ParamSet

    tight = ParamSet(6_000.0, 9.0, 0.8, 6_000.0, 9.0, 0.8, True, 300, 1.0)
    protection.restore_state(replace(protection.state, committed=tight, revision=4))
    status = platform.recalibrate_receiver(9_000.0, 5.0, 1.0, operator="maint-7")
    assert status["revision"] == 5
    assert status["calibrated"]["capacity_kj_per_k"] == 9_000.0
    records = [
        item
        for item in platform.audit.entries(limit=50)
        if item["action"] == "self_params_recalibrated"
    ]
    assert records and records[0]["payload"]["operator"] == "maint-7"


def test_console_exposes_self_params_and_replay(tmp_path):
    from cspplant.console.api import Api

    platform = _noon_platform(tmp_path)
    platform.start()
    api = Api(platform)
    status, payload = api.handle("GET", "/api/self-params", {}, {})
    assert status == 200
    assert payload["self_params"]["schema"] == 2
    assert {item["part"] for item in payload["policy"]["parts"]} == {"panel", "header", "support"}
    status, verdict = api.handle("GET", "/api/protection/verify", {}, {})
    assert status == 200
    assert verdict["ok"] is True

    # 重新标定接口参数非法时返回 4xx（数值解析失败，与既有标定接口同口径）。
    status, bad = api.handle(
        "POST", "/api/receiver/recalibrate", {}, {"capacity_kj_per_k": "abc"}
    )
    assert status in (400, 409)


def test_unknown_self_param_version_fails_closed_to_calibration(tmp_path, clock):
    """落盘了代码无法解释的新版本时：退回纯标定并告警，绝不猜测解释。"""

    first = build_plant(default_config(), clock=clock, storage_root=str(tmp_path / "state"))
    first.store.write(
        "receiver/self_params",
        {
            "schema": 99,
            "revision": 1,
            "calibrated": {"capacity_kj_per_k": 8000.0, "loss_kw_per_k": 6.0},
            "committed": {},
        },
    )
    restarted = build_plant(
        default_config(), clock=clock, storage_root=str(tmp_path / "state")
    )
    # 保护仍可用，但生效值是出厂标定（fail-closed）。
    assert restarted.receiver.protection.state.revision == 0
    assert restarted.receiver.protection.effective_params().capacity_kj_per_k == 8_000.0
    alarms = [
        item
        for item in restarted.audit.entries(limit=20)
        if item["action"] == "self_params_load_failed"
    ]
    assert alarms and alarms[0]["severity"] == "alarm"
