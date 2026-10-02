"""吸热器热工模型。

出口温度由能量平衡决定：入射热功率先加热吸热器金属与熔盐存量，再由流过的熔盐
把热量带走，同时按温差散热。稳态下 ``T_out = T_in + Q / (ṁ·c_p)``，动态过程
由热容给出时间常数——没有热容的话温度会在一秒内跳到稳态，既不真实、也让跳闸
保护失去意义。流量不足时带走的热量变小，温度按同样的时间常数持续爬升，这正是
干烧保护的物理原因。

真值侧的三个参数（实际热容、实际散热、流量计增益）允许在投运后改变，用来模拟
用久了结焦、保温老化与流量计漂移：能量平衡始终按真实值走，而控制器看到的流量
是"指示流量"（真实流量 ÷ 增益），漂移后指示值会高估真实冷却能力。
"""

from __future__ import annotations

from ..config import ReceiverLimits

# 太阳盐在 300–600 ℃ 区间的比热容（kJ/(kg·K)）。
SALT_CP_KJ_KG_K = 1.53

# 冷盐进吸热器的温度（℃），由冷盐罐侧给出。
INLET_TEMP_C = 290.0

# 吸热器金属管屏与在管熔盐的等效热容（kJ/K）。
RECEIVER_CAPACITY_KJ_K = 8_000.0

# 按温差的散热系数（kW/K）。
LOSS_KW_K = 6.0


class ThermalModel:
    """按热功率与流量推算出口熔盐温度。"""

    def __init__(
        self,
        limits: ReceiverLimits,
        *,
        capacity_kj_per_k: float = RECEIVER_CAPACITY_KJ_K,
        loss_kw_per_k: float = LOSS_KW_K,
        flow_gain: float = 1.0,
    ) -> None:
        if capacity_kj_per_k <= 0:
            raise ValueError("热容必须为正")
        if loss_kw_per_k < 0:
            raise ValueError("散热系数不得为负")
        if not 0.0 < flow_gain <= 2.0:
            raise ValueError("流量计增益必须落在 (0, 2]")
        self._limits = limits
        self._inlet_c = INLET_TEMP_C
        self._temp_c = float(limits.ambient_c)
        self._updates = 0
        self._peak_c = self._temp_c
        self._dry_steps = 0
        self._capacity = float(capacity_kj_per_k)
        self._loss = float(loss_kw_per_k)
        self._flow_gain = float(flow_gain)

    @property
    def limits(self) -> ReceiverLimits:
        return self._limits

    @property
    def inlet_c(self) -> float:
        """返回冷盐入口温度。"""

        return self._inlet_c

    @property
    def capacity_kj_per_k(self) -> float:
        """返回实际等效热容，决定温升的时间常数。"""

        return self._capacity

    @property
    def loss_kw_per_k(self) -> float:
        """返回实际散热系数（真值）。"""

        return self._loss

    @property
    def flow_gain(self) -> float:
        """返回实际流量计增益：真实流量 = 指示流量 × 增益。"""

        return self._flow_gain

    def degrade(
        self,
        *,
        capacity_kj_per_k: float | None = None,
        loss_kw_per_k: float | None = None,
        flow_gain: float | None = None,
    ) -> None:
        """注入一次真值侧漂移（结焦／保温老化／流量计漂移），仅供仿真与验证。"""

        if capacity_kj_per_k is not None:
            if capacity_kj_per_k <= 0:
                raise ValueError("热容必须为正")
            self._capacity = float(capacity_kj_per_k)
        if loss_kw_per_k is not None:
            if loss_kw_per_k < 0:
                raise ValueError("散热系数不得为负")
            self._loss = float(loss_kw_per_k)
        if flow_gain is not None:
            if not 0.0 < flow_gain <= 2.0:
                raise ValueError("流量计增益必须落在 (0, 2]")
            self._flow_gain = float(flow_gain)

    def reset(self, temperature_c: float | None = None) -> float:
        """把温度复位到给定值（默认环境温度）。"""

        self._temp_c = float(self._limits.ambient_c if temperature_c is None else temperature_c)
        self._peak_c = self._temp_c
        return self._temp_c

    def update(self, power_kw: float, flow_kg_s: float, dt: float) -> float:
        """推进一个步长，返回推进后的出口温度。

        传入的是流量计的**指示流量**；能量平衡按真实流量（指示值×增益）
        记账，干烧计数仍按指示值与联锁门限比较（联锁只看得到读数）。
        """

        if dt <= 0:
            raise ValueError("步长必须为正")
        indicated = max(0.0, float(flow_kg_s))
        if indicated < self._limits.flow_min_kg_s:
            self._dry_steps += 1
        real_flow = indicated * self._flow_gain
        absorbed = float(power_kw) * dt
        carried = real_flow * SALT_CP_KJ_KG_K * (self._temp_c - self._inlet_c) * dt
        lost = self._loss * (self._temp_c - self._limits.ambient_c) * dt
        self._temp_c += (absorbed - carried - lost) / self._capacity
        self._temp_c = max(self._limits.ambient_c, self._temp_c)
        self._peak_c = max(self._peak_c, self._temp_c)
        self._updates += 1
        return self._temp_c

    def value_c(self) -> float:
        """返回当前出口温度。"""

        return self._temp_c

    def peak_c(self) -> float:
        """返回历史最高出口温度。"""

        return self._peak_c

    def updates(self) -> int:
        """返回累计推进步数。"""

        return self._updates

    def dry_steps(self) -> int:
        """返回流量不足的推进步数。"""

        return self._dry_steps

    def derate_factor(self) -> float:
        """返回按出口温度给出的出力系数（1.0 表示不降额）。

        这是按固定标定阈值的旧口径，仅在未挂版本化保护模型时兜底；
        投运保护以 :class:`~cspplant.receiver.protection.ProtectionModel`
        的判定为准。
        """

        if self._temp_c < self._limits.temp_derate_c:
            return 1.0
        span = self._limits.temp_max_c - self._limits.temp_derate_c
        if span <= 0:
            return 0.0
        return max(0.0, min(1.0, (self._limits.temp_max_c - self._temp_c) / span))
