"""热容、散热系数与流量计增益的在线辨识。

能量平衡（C 为等效热容、UA 为散热系数、g 为流量计增益，真实流量=指示
流量×g）：

    C·dT/dt = P − g·cp·ṁ·(T−T_in) − UA·(T−T_amb)

两边除以 C 后对 [1/C, g·cp/C, UA/C] 是线性的，可以做带遗忘因子的递推
最小二乘（RLS）。为避免 P、流量、温升各数量级差几千倍把矩阵跑坏，先按
标定点把回归量归一化，辨识三个无量纲系数 a、b、d（标定时都等于 1）：

    a·v_P − b·v_flow − d·v_loss = y

恢复物理量：C=C0/a，g=g0·b/a，UA=UA0·d/a。

**可辨识性按维度独立判定**：流量项与散热项都正比于温差，流量恒定时二者
严格共线，任何算法都分不开 g 与 UA。此时只允许采纳热容（功率方向始终有
激励），g、UA 保持标定，绝不能把不可分维度上的虚假估计写进保护。

辨识器**只产原始估计与质量标志**，不决定保护怎么用：某维不可辨识或整体
拟合残差过大时，对应标志为假，上层不得据此收紧或放松该维保护值。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .self_params import ParamSet

# 太阳盐 300–600 ℃ 比热容 kJ/(kg·K)，与热工模型同一口径。
SALT_CP_KJ_KG_K = 1.53

# 遗忘因子：越小越跟手，但越容易被短期噪声带跑。
FORGETTING = 0.995

# 初始协方差（无量纲系数空间，先验为标定点 [1,1,1]）。
INITIAL_COV = 100.0

# 最小样本数、最大拟合残差（占标定热功率的比例）。
MIN_SAMPLES = 60
MAX_RESIDUAL_RATIO = 0.01

# 各维度可辨识性：激励 gram 的对应特征值必须超过这个门槛。
# 热容对应主功率方向，量级最大；流量/散热方向天然偏弱，门限相应放低。
MIN_EIG_CAPACITY = 0.05
MIN_EIG_COOLING = 1e-5

# 残差窗口长度（步数）。
RESIDUAL_WINDOW = 120


@dataclass(frozen=True)
class Estimate:
    """一次辨识输出。标志为假的维度不能被保护采纳。"""

    raw: ParamSet
    residual_c_s: float
    residual_ratio: float
    eigenvalues: tuple[float, float, float]
    capacity_identifiable: bool
    cooling_identifiable: bool
    started: bool

    @property
    def fit_ok(self) -> bool:
        """至少热容可辨识且整体残差达标（各维标志另由属性判断）。"""

        return self.raw.fit_ok


def _eigenvalues_3x3(matrix: tuple[tuple[float, float, float], ...]) -> tuple[float, float, float]:
    """用 Jacobi 旋转求 3×3 对称阵的全部特征值（标准库实现，避免三方依赖）。"""

    a = [list(row) for row in matrix]
    for _ in range(24):
        off = max(abs(a[0][1]), abs(a[0][2]), abs(a[1][2]))
        if off < 1e-14:
            break
        for p, q in ((0, 1), (0, 2), (1, 2)):
            apq = a[p][q]
            if abs(apq) < 1e-16:
                continue
            tau = (a[q][q] - a[p][p]) / (2.0 * apq)
            t = 1.0 / (tau + (1.0 if tau >= 0 else -1.0) * math.sqrt(1.0 + tau * tau)) if tau else 1.0
            c = 1.0 / math.sqrt(1.0 + t * t)
            s = t * c
            for k in range(3):
                akp, akq = a[k][p], a[k][q]
                a[k][p] = c * akp - s * akq
                a[k][q] = s * akp + c * akq
            for k in range(3):
                apk, aqs = a[p][k], a[q][k]
                a[p][k] = c * apk - s * aqs
                a[q][k] = s * apk + c * aqs
    return a[0][0], a[1][1], a[2][2]


class ThermalEstimator:
    """按实测功率、指示流量与温升在线辨识热工参数。"""

    def __init__(
        self,
        calibrated,
        *,
        inlet_c: float,
        ambient_c: float,
        cp_kj_kg_k: float = SALT_CP_KJ_KG_K,
        forgetting: float = FORGETTING,
    ) -> None:
        self._cal = calibrated
        self._inlet_c = float(inlet_c)
        self._ambient_c = float(ambient_c)
        self._cp = float(cp_kj_kg_k)
        self._lam = float(forgetting)
        # 标定特征：60 kg/s、300 K 温升下带走的热功率，作为统一归一化尺度。
        self._s_power = max(1.0, calibrated.flow_gain * self._cp * 60.0 * 300.0)
        self.reset()

    def reset(self) -> None:
        """复位辨识状态（先验回到标定点）。"""

        # 无量纲系数 [a, b, d]，标定点先验为 [1, 1, 1]。
        self._theta = [1.0, 1.0, 1.0]
        self._cov = [
            [INITIAL_COV, 0.0, 0.0],
            [0.0, INITIAL_COV, 0.0],
            [0.0, 0.0, INITIAL_COV],
        ]
        self._last_temp: float | None = None
        self._samples = 0
        self._residuals: list[float] = []
        self._gram = [[0.0] * 3 for _ in range(3)]
        self._last_residual_c_s = 0.0
        self._last_eigs: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_flow = 0.0
        self._flow_seen: list[float] = []

    @property
    def samples(self) -> int:
        return self._samples

    def _regressor(self, power_kw: float, flow_indicated_kg_s: float, temp_c: float) -> tuple[float, ...]:
        """构造无量纲回归向量 v。归一化方程：a·v0 − b·v1 − d·v2 = y。"""

        carried = self._cal.flow_gain * self._cp * max(0.0, float(flow_indicated_kg_s)) * (
            temp_c - self._inlet_c
        )
        lost = self._cal.loss_kw_per_k * (temp_c - self._ambient_c)
        return (
            float(power_kw) / self._s_power,
            -(carried / self._s_power),
            -(lost / self._s_power),
        )

    def _identifiability(self) -> tuple[bool, bool, tuple[float, float, float]]:
        """按激励 gram 的特征值判定各维度可辨识性。

        最大特征值方向是功率（热容）；两个较小特征值对应流量与散热方向。
        流量是否真正参与激励，还要看流量本身有没有变化（恒定流量下流量项
        只是散热项的固定倍数）。
        """

        eigs = tuple(sorted(_eigenvalues_3x3(tuple(tuple(row) for row in self._gram)), reverse=True))
        capacity_ok = self._samples >= MIN_SAMPLES and eigs[0] >= MIN_EIG_CAPACITY
        flow_varied = bool(self._flow_seen) and (max(self._flow_seen) - min(self._flow_seen)) > 2.0
        cooling_ok = self._samples >= MIN_SAMPLES and eigs[1] >= MIN_EIG_COOLING and (
            flow_varied or eigs[2] >= MIN_EIG_COOLING
        )
        # 流量没变化时，g 与 UA 只能合起来辨识；此时把散热维也标为不可单独
        # 采纳，保护对两维都保持标定（它们的合量已体现在热容残差里）。
        if not flow_varied:
            cooling_ok = False
        return capacity_ok, cooling_ok, eigs

    def observe(
        self,
        power_kw: float,
        flow_indicated_kg_s: float,
        raw_temp_c: float,
        dt: float,
    ) -> Estimate:
        """推进一个采样步，返回当前原始估计。

        温度必须用传感器滤波前的原始值做物理辨识（滤波会引入相位滞后，
        相当于虚增热容）；保护门限判定仍在别处使用滤波后的稳态读数。
        """

        if dt <= 0:
            raise ValueError("步长必须为正")
        temp = float(raw_temp_c)
        self._last_flow = max(0.0, float(flow_indicated_kg_s))
        if len(self._flow_seen) < RESIDUAL_WINDOW:
            self._flow_seen.append(self._last_flow)
        else:
            self._flow_seen.pop(0)
            self._flow_seen.append(self._last_flow)
        if self._last_temp is None:
            self._last_temp = temp
            return self._estimate(started=False)

        dtemp_c_s = (temp - self._last_temp) / float(dt)
        self._last_temp = temp
        v = self._regressor(power_kw, self._last_flow, temp)
        y = self._cal.capacity_kj_per_k * dtemp_c_s / self._s_power
        prediction = sum(self._theta[i] * v[i] for i in range(3))
        innovation = y - prediction
        self._rls_update(v, innovation)

        for i in range(3):
            for j in range(3):
                self._gram[i][j] = 0.98 * self._gram[i][j] + 0.02 * v[i] * v[j]
        self._samples += 1
        self._last_residual_c_s = innovation * self._s_power / self._cal.capacity_kj_per_k
        self._residuals.append(abs(self._last_residual_c_s))
        if len(self._residuals) > RESIDUAL_WINDOW:
            self._residuals.pop(0)
        self._last_eigs = _eigenvalues_3x3(tuple(tuple(row) for row in self._gram))
        return self._estimate(started=True)

    def _rls_update(self, v: tuple[float, ...], innovation: float) -> None:
        """标准带遗忘因子 RLS 协方差更新（无量纲空间）。"""

        pv = [sum(self._cov[i][j] * v[j] for j in range(3)) for i in range(3)]
        denom = self._lam + sum(v[i] * pv[i] for i in range(3))
        gain = [value / denom for value in pv]
        for i in range(3):
            self._theta[i] += gain[i] * innovation
            for j in range(3):
                self._cov[i][j] = (self._cov[i][j] - gain[i] * pv[j]) / self._lam
        # 系数必须为正才有物理意义；投影回正区间，防止坏数据把恢复量算飞。
        self._theta = [max(1e-4, value) for value in self._theta]

    def _recover(self) -> tuple[float, float, float]:
        a, b, d = self._theta
        capacity = self._cal.capacity_kj_per_k / a
        gain = self._cal.flow_gain * b / a
        loss = self._cal.loss_kw_per_k * d / a
        return capacity, loss, gain

    def _estimate(self, *, started: bool) -> Estimate:
        capacity, loss, gain = self._recover()
        capacity_ok, cooling_ok, eigs = self._identifiability()
        residual_c_s = sum(self._residuals) / len(self._residuals) if self._residuals else 0.0
        residual_ratio = abs(residual_c_s) * self._cal.capacity_kj_per_k / self._s_power
        residual_ok = residual_ratio <= MAX_RESIDUAL_RATIO
        plausible = (
            1_000.0 <= capacity <= 100_000.0
            and 0.0 <= loss <= 200.0
            and 0.5 <= gain <= 1.5
        )
        capacity_adoptable = bool(started and capacity_ok and residual_ok and plausible)
        cooling_adoptable = bool(started and cooling_ok and residual_ok and plausible)
        # fit_ok 表示"至少有可采纳的维度"，整体仍带各维独立标志。
        raw = ParamSet(
            capacity_kj_per_k=capacity if capacity_adoptable else self._cal.capacity_kj_per_k,
            loss_kw_per_k=loss if cooling_adoptable else self._cal.loss_kw_per_k,
            flow_gain=gain if cooling_adoptable else self._cal.flow_gain,
            raw_capacity_kj_per_k=capacity,
            raw_loss_kw_per_k=loss,
            raw_flow_gain=gain,
            fit_ok=capacity_adoptable or cooling_adoptable,
            samples=self._samples,
            quality=max(0.0, min(1.0, eigs[0] / max(MIN_EIG_CAPACITY, 1e-9))),
        )
        return Estimate(
            raw=raw,
            residual_c_s=residual_c_s,
            residual_ratio=residual_ratio,
            eigenvalues=eigs,
            capacity_identifiable=capacity_adoptable,
            cooling_identifiable=cooling_adoptable,
            started=started,
        )
