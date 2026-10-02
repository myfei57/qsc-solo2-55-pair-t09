"""在线辨识器：用已知真值的合成热工过程验证收敛与质量门。"""

from __future__ import annotations

import math

from cspplant.receiver.estimator import ThermalEstimator
from cspplant.receiver.self_params import CalibratedParams

CP = 1.53
TIN = 290.0
TAMB = 22.0
CAL = CalibratedParams(8_000.0, 6.0, 1.0)


def _drive(estimator, *, c_true, ua_true, g_true, steps, vary_flow=True, seed=1):
    t = 300.0
    dt = 2.0
    # 固定序列的伪噪声，保证测试可复现。
    noise = [((n * 37) % 11 - 5) * 0.02 for n in range(steps)]
    for i in range(steps):
        power = 18_000 + 12_000 * math.sin(i / 25.0) + 4_000 * math.sin(i / 7.0)
        flow = 58.0 + (8.0 * math.sin(i / 40.0) if vary_flow else 0.0)
        real_flow = flow * g_true
        dtemp = (power - real_flow * CP * (t - TIN) - ua_true * (t - TAMB)) / c_true
        t += dtemp * dt + noise[i % len(noise)]
        estimator.observe(power, flow, t, dt)
    return estimator._estimate(started=True)


def test_estimator_recovers_capacity_loss_and_gain_under_excitation():
    estimator = ThermalEstimator(CAL, inlet_c=TIN, ambient_c=TAMB)
    result = _drive(estimator, c_true=6_000.0, ua_true=9.0, g_true=0.8, steps=1_200)
    raw = result.raw
    assert result.capacity_identifiable is True
    assert result.cooling_identifiable is True
    # 辨识误差落在工程可接受带内，且热容估计偏保守（不大于真值太多）。
    assert 5_400.0 < raw.raw_capacity_kj_per_k < 6_300.0
    assert abs(raw.raw_flow_gain - 0.8) < 0.05
    assert abs(raw.raw_loss_kw_per_k - 9.0) < 1.0


def test_cooling_dimensions_not_adopted_when_flow_is_constant():
    estimator = ThermalEstimator(CAL, inlet_c=TIN, ambient_c=TAMB)
    result = _drive(estimator, c_true=4_000.0, ua_true=6.0, g_true=1.0, steps=1_200, vary_flow=False)
    # 功率方向始终有激励，热容可辨识（4000 < 标定 8000）。
    assert result.capacity_identifiable is True
    assert result.raw.capacity_kj_per_k < 8_000.0
    # 流量恒定，g 与 UA 物理不可分，禁止采纳。
    assert result.cooling_identifiable is False
    assert result.raw.flow_gain == 1.0
    assert result.raw.loss_kw_per_k == 6.0


def test_first_observe_is_warmup_without_commit():
    estimator = ThermalEstimator(CAL, inlet_c=TIN, ambient_c=TAMB)
    first = estimator.observe(20_000.0, 58.0, 300.0, 2.0)
    assert first.started is False
    assert first.fit_ok is False
    assert estimator.samples == 0
