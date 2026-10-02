"""吸热器热工参数自算与版本化降额保护。

固定热容、固定散热系数在结焦与流量计漂移之后会越来越偏：同样工况下保护有时
提前、有时滞后。这里用每个受热步的功率、流量与温升反解能量平衡，在线估计
等效热容与散热系数，并按两条铁律使用估计结果：

1. 只能更保守——热容与散热只准向下迁移（结焦让温升更快、散热更差），乐观
   估计逐项钳回当前值；迁移还有下限，防止坏数据把保护打到不可用。
2. 来源定死——每个保护部位按配置里的裁决矩阵选用自算值还是标定值，运行
   时没有开关；超温跳闸门限永远沿用标定，自算只能影响降额起点与出力上限。

每次采纳新参数都是一次迁移：版本号加一、全量落盘、写审计。实时判定、回放
与审计按同一个版本号解释同一组参数；回放时按记录里的版本重算降额边界，与
当时的记录逐字段核对，不一致即报。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable

from ..config import AdaptiveLimits, ReceiverLimits
from ..store.jsonfile import JsonStore
from .derate import HYSTERESIS_C
from .thermal import (
    INLET_TEMP_C,
    LOSS_KW_K,
    RECEIVER_CAPACITY_KJ_K,
    SALT_CP_KJ_KG_K,
)

LEARNED = "learned"
CALIBRATION = "calibration"

# 自算参数版本库的落盘文档标识。
STORE_DOC_ID = "receiver/adaptive_params"

# 回放重算允许的误差（℃ / %）：落盘舍入之内的差异不算不一致。
REPLAY_TOLERANCE = 0.05


def _ceiling_interp(limits: ReceiverLimits, temperature_c: float, derate_start_c: float) -> float:
    """按降额起点线性插值出力上限（纯函数，实时与回放共用同一套）。"""

    temp = float(temperature_c)
    if temp <= derate_start_c:
        return 100.0
    span = limits.temp_max_c - derate_start_c
    if span <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * (limits.temp_max_c - temp) / span))


@dataclass(frozen=True)
class ThermalEstimate:
    """一次窗口化能量平衡反解的结果（未钳制的原始估计）。"""

    capacity_kj_k: float
    loss_kw_k: float
    samples: int
    temp_span_c: float
    power_swing_kw: float
    residual_ratio: float
    created_ts: float

    def as_dict(self) -> dict:
        return {
            "capacity_kj_k": round(self.capacity_kj_k, 1),
            "loss_kw_k": round(self.loss_kw_k, 4),
            "samples": self.samples,
            "temp_span_c": round(self.temp_span_c, 2),
            "power_swing_kw": round(self.power_swing_kw, 1),
            "residual_ratio": round(self.residual_ratio, 4),
            "created_ts": round(self.created_ts, 3),
        }


class ThermalEstimator:
    """按能量平衡从功率、流量与温升反解等效热容与散热系数。

    每个受热步满足 ``C·ΔT = P·dt − ṁ·cp·(T−T_in)·dt − h·(T−T_env)·dt``，
    把 ``C`` 与 ``h`` 当作未知量，对窗口内的样本解二元最小二乘。窗口内温度
    与功率都没有足够摆动时两个未知量不可分，估计器保持沉默——病态方程的解
    不允许去驱动保护。拟合残差超限时同样保持沉默：窗口横跨工况突变（结焦、
    流量计跳漂）时单一参数组拟合两段真值必然失真，而失真估计一旦被单调
    钳制固化就再也无法回上调。
    """

    def __init__(self, limits: ReceiverLimits, adaptive: AdaptiveLimits, inlet_c: float) -> None:
        self._limits = limits
        self._adaptive = adaptive
        self._inlet = float(inlet_c)
        self._samples: deque[tuple[float, float, float, float, float]] = deque(
            maxlen=adaptive.window
        )
        self._fed = 0
        self._since_estimate = 0

    def feed(
        self,
        power_kw: float,
        flow_kg_s: float,
        temp_prev_c: float,
        temp_now_c: float,
        dt: float,
    ) -> None:
        """喂入一个受热步；低流量样本不参与（干烧段会污染能量平衡）。"""

        if dt <= 0:
            return
        flow = float(flow_kg_s)
        if flow < self._limits.flow_min_kg_s:
            return
        prev = float(temp_prev_c)
        x1 = float(temp_now_c) - prev
        x2 = (prev - self._limits.ambient_c) * float(dt)
        y = (float(power_kw) - flow * SALT_CP_KJ_KG_K * (prev - self._inlet)) * float(dt)
        self._samples.append((prev, float(power_kw), x1, x2, y))
        self._fed += 1
        self._since_estimate += 1

    def samples(self) -> int:
        """返回当前窗口内的样本数。"""

        return len(self._samples)

    def fed(self) -> int:
        """返回累计喂入的样本数。"""

        return self._fed

    def ready(self) -> bool:
        """窗口已满且距上次估计又积累了半个窗口的新样本。"""

        return (
            len(self._samples) >= self._adaptive.window
            and self._since_estimate >= max(1, self._adaptive.window // 2)
        )

    def estimate(self, now: float) -> ThermalEstimate | None:
        """解一次窗口最小二乘；激励不足或解不合法时返回 ``None``。"""

        if not self.ready():
            return None
        self._since_estimate = 0
        points = list(self._samples)
        s11 = sum(p[2] * p[2] for p in points)
        s12 = sum(p[2] * p[3] for p in points)
        s22 = sum(p[3] * p[3] for p in points)
        b1 = sum(p[2] * p[4] for p in points)
        b2 = sum(p[3] * p[4] for p in points)
        det = s11 * s22 - s12 * s12
        if s11 <= 0.0 or s22 <= 0.0 or det <= 1e-6 * s11 * s22:
            return None
        capacity = (b1 * s22 - b2 * s12) / det
        loss = (s11 * b2 - s12 * b1) / det
        if not (math.isfinite(capacity) and math.isfinite(loss)):
            return None
        if capacity <= 0.0 or loss <= 0.0:
            return None
        temps = [p[0] for p in points]
        temps.append(points[-1][0] + points[-1][2])
        temp_span = max(temps) - min(temps)
        powers = [p[1] for p in points]
        power_swing = max(powers) - min(powers)
        if (
            temp_span < self._adaptive.min_temp_rise_c
            and power_swing < self._adaptive.min_power_swing_kw
        ):
            return None
        sse = 0.0
        syy = 0.0
        for p in points:
            residual = capacity * p[2] + loss * p[3] - p[4]
            sse += residual * residual
            syy += p[4] * p[4]
        residual_ratio = math.sqrt(sse / syy) if syy > 0.0 else 0.0
        # 拟合残差超限时保持沉默：窗口横跨结焦等工况突变时，单一 (C, h) 拟合
        # 两段真值必然失真，这种解一旦经单调钳制固化就再也无法回上调。
        if residual_ratio > self._adaptive.max_residual_ratio:
            return None
        return ThermalEstimate(
            capacity_kj_k=capacity,
            loss_kw_k=loss,
            samples=len(points),
            temp_span_c=temp_span,
            power_swing_kw=power_swing,
            residual_ratio=residual_ratio,
            created_ts=float(now),
        )


@dataclass(frozen=True)
class ParameterVersion:
    """一版自算热工参数及其派生的降额边界。

    降额起点与出力上限在迁移时由热容、散热推导并随版本一起落盘；回放按同一
    推导重算，保证同一版本号在任何时刻解释出同一组边界。
    """

    version: int
    source: str
    capacity_kj_k: float
    loss_kw_k: float
    derate_start_c: float
    ceiling_cap_pct: float
    created_ts: float
    samples: int = 0
    residual_ratio: float = 0.0
    clamped: bool = False
    estimate_capacity_kj_k: float = 0.0
    estimate_loss_kw_k: float = 0.0

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "source": self.source,
            "capacity_kj_k": round(self.capacity_kj_k, 3),
            "loss_kw_k": round(self.loss_kw_k, 4),
            "derate_start_c": round(self.derate_start_c, 3),
            "ceiling_cap_pct": round(self.ceiling_cap_pct, 3),
            "created_ts": round(self.created_ts, 3),
            "samples": self.samples,
            "residual_ratio": round(self.residual_ratio, 4),
            "clamped": self.clamped,
            "estimate_capacity_kj_k": round(self.estimate_capacity_kj_k, 3),
            "estimate_loss_kw_k": round(self.estimate_loss_kw_k, 4),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "ParameterVersion":
        """从落盘字段还原一版参数。"""

        return cls(
            version=int(raw.get("version", 0)),
            source=str(raw.get("source", CALIBRATION)),
            capacity_kj_k=float(raw.get("capacity_kj_k", RECEIVER_CAPACITY_KJ_K)),
            loss_kw_k=float(raw.get("loss_kw_k", LOSS_KW_K)),
            derate_start_c=float(raw.get("derate_start_c", 0.0)),
            ceiling_cap_pct=float(raw.get("ceiling_cap_pct", 100.0)),
            created_ts=float(raw.get("created_ts", 0.0)),
            samples=int(raw.get("samples", 0)),
            residual_ratio=float(raw.get("residual_ratio", 0.0)),
            clamped=bool(raw.get("clamped", False)),
            estimate_capacity_kj_k=float(raw.get("estimate_capacity_kj_k", 0.0)),
            estimate_loss_kw_k=float(raw.get("estimate_loss_kw_k", 0.0)),
        )


@dataclass(frozen=True)
class MigrationResult:
    """一次迁移尝试的结果。"""

    migrated: bool
    reason: str
    version: ParameterVersion | None
    clamped: bool

    def as_dict(self) -> dict:
        payload: dict = {"migrated": self.migrated, "reason": self.reason, "clamped": self.clamped}
        if self.version is not None:
            payload["version"] = self.version.as_dict()
        return payload


class ParameterStore:
    """自算参数的版本库：迁移只准更保守，全量落盘供重启恢复与回放。

    迁移规则（单调保守）：候选热容、散热逐项取 ``min(估计值, 当前值)``，再
    钳到标定值乘以各自下限比例之上；钳制后相对当前版本的保守收益低于迁移
    门槛时不改版，避免测量噪声把版本号刷得失去意义。
    """

    def __init__(
        self,
        limits: ReceiverLimits,
        adaptive: AdaptiveLimits,
        *,
        reference_flow_kg_s: float,
        capacity_cal: float = RECEIVER_CAPACITY_KJ_K,
        loss_cal: float = LOSS_KW_K,
        persistence: JsonStore | None = None,
        doc_id: str = STORE_DOC_ID,
    ) -> None:
        self._limits = limits
        self._adaptive = adaptive
        self._reference_flow = float(reference_flow_kg_s)
        self._capacity_cal = float(capacity_cal)
        self._loss_cal = float(loss_cal)
        self._store = persistence
        self._doc_id = doc_id
        self._versions: list[ParameterVersion] = []
        self._rejections = 0
        self._last_rejection = ""
        self._restore()

    @property
    def limits(self) -> ReceiverLimits:
        return self._limits

    def _restore(self) -> None:
        if self._store is not None:
            raw = self._store.read(self._doc_id)
            if isinstance(raw, dict) and isinstance(raw.get("versions"), list) and raw["versions"]:
                versions = [ParameterVersion.from_dict(item) for item in raw["versions"]]
                if versions[0].version == 0:
                    self._versions = versions
                    return
        self._versions = [self._calibration_version()]
        self._persist()

    def _persist(self) -> None:
        if self._store is None:
            return
        self._store.write(
            self._doc_id,
            {
                "current": self.current().version,
                "versions": [version.as_dict() for version in self._versions],
            },
        )

    def _calibration_version(self) -> ParameterVersion:
        start, cap = self._derive(self._capacity_cal, self._loss_cal)
        return ParameterVersion(
            version=0,
            source=CALIBRATION,
            capacity_kj_k=self._capacity_cal,
            loss_kw_k=self._loss_cal,
            derate_start_c=start,
            ceiling_cap_pct=cap,
            created_ts=0.0,
        )

    def _derive(self, capacity_kj_k: float, loss_kw_k: float) -> tuple[float, float]:
        """由热容与散热推导降额起点与出力上限（纯函数，回放用同一套）。

        热容下降 → 同样功率温升更快 → 降额起点向目标温度靠拢（提前）；散热
        下降 → 不超温可持续的功率变小 → 出力上限压低。两个方向都只会更保守。
        """

        ratio = max(0.0, min(1.0, float(capacity_kj_k) / self._capacity_cal))
        start = self._limits.temp_target_c + (self._limits.temp_derate_c - self._limits.temp_target_c) * ratio
        head = self._limits.temp_max_c - self._limits.ambient_c
        carried = self._reference_flow * SALT_CP_KJ_KG_K * (self._limits.temp_max_c - INLET_TEMP_C)
        q_cal = carried + self._loss_cal * head
        q_est = carried + float(loss_kw_k) * head
        cap = 100.0 * q_est / q_cal if q_cal > 0.0 else 100.0
        return start, max(0.0, min(100.0, cap))

    def derive(self, version: ParameterVersion) -> tuple[float, float]:
        """重算一版参数的降额边界，供回放核对。"""

        return self._derive(version.capacity_kj_k, version.loss_kw_k)

    def calibration(self) -> ParameterVersion:
        """返回标定版本（v0）。"""

        return self._versions[0]

    def current(self) -> ParameterVersion:
        """返回当前生效版本。"""

        return self._versions[-1]

    def get(self, version: int) -> ParameterVersion | None:
        """按版本号取参数；不存在时返回 ``None``。"""

        for item in self._versions:
            if item.version == version:
                return item
        return None

    def versions(self) -> list[ParameterVersion]:
        """返回全部版本（含标定 v0）。"""

        return list(self._versions)

    def rejections(self) -> int:
        """返回累计被拒绝的迁移次数。"""

        return self._rejections

    def last_rejection(self) -> str:
        """返回最近一次拒绝的原因。"""

        return self._last_rejection

    def propose(self, estimate: ThermalEstimate, now: float) -> MigrationResult:
        """尝试把一次估计迁移成新版本；只接受更保守的方向。"""

        current = self.current()
        floor_capacity = self._capacity_cal * self._adaptive.capacity_floor_ratio
        floor_loss = self._loss_cal * self._adaptive.loss_floor_ratio
        capacity = max(min(estimate.capacity_kj_k, current.capacity_kj_k), floor_capacity)
        loss = max(min(estimate.loss_kw_k, current.loss_kw_k), floor_loss)
        clamped = capacity != estimate.capacity_kj_k or loss != estimate.loss_kw_k
        margin = self._adaptive.migration_margin_pct / 100.0
        gain = max(
            1.0 - capacity / current.capacity_kj_k,
            1.0 - loss / current.loss_kw_k,
        )
        if gain < margin:
            self._rejections += 1
            self._last_rejection = "估计结果不比当前版本更保守（低于迁移门槛）"
            return MigrationResult(False, "gain_below_margin", None, clamped)
        start, cap = self._derive(capacity, loss)
        version = ParameterVersion(
            version=current.version + 1,
            source=LEARNED,
            capacity_kj_k=capacity,
            loss_kw_k=loss,
            derate_start_c=start,
            ceiling_cap_pct=cap,
            created_ts=float(now),
            samples=estimate.samples,
            residual_ratio=estimate.residual_ratio,
            clamped=clamped,
            estimate_capacity_kj_k=estimate.capacity_kj_k,
            estimate_loss_kw_k=estimate.loss_kw_k,
        )
        self._versions.append(version)
        self._persist()
        return MigrationResult(True, "migrated", version, clamped)


@dataclass(frozen=True)
class AdaptiveVerdict:
    """一次降额判定结果，携带解释所依据的版本与参数。"""

    zone: str
    source: str
    version: int
    active: bool
    ceiling_pct: float
    derate_start_c: float
    ceiling_cap_pct: float
    temperature_c: float
    changed: bool
    explanation: str

    def as_dict(self) -> dict:
        return {
            "zone": self.zone,
            "source": self.source,
            "version": self.version,
            "active": self.active,
            "ceiling_pct": round(self.ceiling_pct, 3),
            "derate_start_c": round(self.derate_start_c, 3),
            "ceiling_cap_pct": round(self.ceiling_cap_pct, 3),
            "temperature_c": round(self.temperature_c, 2),
            "changed": self.changed,
            "explanation": self.explanation,
        }


class AdaptiveProtection:
    """按部位裁决矩阵给出降额判定；来源定死，参数按版本解释。

    每次判定都与上一次对比并生成解释：版本切换时给出参数差（这次保护凭什么
    比上次早），版本不变时明示与上次一致。判定结果是否发生实质变化由
    ``changed`` 标出，平台据此把变化点落审计，回放再按版本核对。
    """

    def __init__(
        self,
        limits: ReceiverLimits,
        adaptive: AdaptiveLimits,
        store: ParameterStore,
        estimator: ThermalEstimator | None = None,
    ) -> None:
        self._limits = limits
        self._adaptive = adaptive
        self._store = store
        self._estimator = estimator
        self._active = False
        self._last_temp = float(limits.temp_target_c)
        self._last_verdict: AdaptiveVerdict | None = None
        self._pending: ThermalEstimate | None = None

    @property
    def store(self) -> ParameterStore:
        return self._store

    @property
    def estimator(self) -> ThermalEstimator | None:
        return self._estimator

    def zone_source(self, zone: str) -> str:
        """返回某保护部位定死的参数来源。"""

        return str(self._adaptive.zone_sources[zone])

    def _params(self) -> ParameterVersion:
        if self.zone_source("panel") == LEARNED:
            return self._store.current()
        return self._store.calibration()

    def feed(
        self,
        *,
        power_kw: float,
        flow_kg_s: float,
        temp_prev_c: float,
        temp_now_c: float,
        dt: float,
        now: float,
    ) -> None:
        """喂入一个受热步的原始量测（功率、流量、未滤波的温升）。"""

        if self._estimator is not None:
            self._estimator.feed(power_kw, flow_kg_s, temp_prev_c, temp_now_c, dt)
            estimate = self._estimator.estimate(now)
            if estimate is not None:
                self._pending = estimate

    def judge(self, temperature_c: float) -> AdaptiveVerdict:
        """按（滤波后的）出口温度给出本次降额判定。"""

        return self._judge(float(temperature_c))

    def _judge(self, temperature_c: float) -> AdaptiveVerdict:
        params = self._params()
        source = self.zone_source("panel")
        start = params.derate_start_c
        cap = params.ceiling_cap_pct
        ceiling = min(_ceiling_interp(self._limits, temperature_c, start), cap)
        toggled = False
        if not self._active and temperature_c >= start:
            self._active = True
            toggled = True
        elif self._active and temperature_c < start - HYSTERESIS_C:
            self._active = False
            toggled = True
        self._last_temp = temperature_c
        previous = self._last_verdict
        changed = (
            previous is None
            or previous.version != params.version
            or toggled
            or abs(previous.ceiling_pct - ceiling) > 0.5
        )
        verdict = AdaptiveVerdict(
            zone="panel",
            source=source,
            version=params.version,
            active=self._active,
            ceiling_pct=ceiling,
            derate_start_c=start,
            ceiling_cap_pct=cap,
            temperature_c=temperature_c,
            changed=changed,
            explanation=self._explain(params, previous, temperature_c, toggled),
        )
        self._last_verdict = verdict
        return verdict

    def _explain(
        self,
        params: ParameterVersion,
        previous: AdaptiveVerdict | None,
        temperature_c: float,
        toggled: bool,
    ) -> str:
        source_label = "自算" if self.zone_source("panel") == LEARNED else "标定"
        if previous is None:
            text = (
                f"首次判定：{source_label} v{params.version}，"
                f"降额起点 {params.derate_start_c:.1f}℃，上限 {params.ceiling_cap_pct:.1f}%"
            )
        elif previous.version != params.version:
            old = self._store.get(previous.version) or params
            earlier = previous.derate_start_c - params.derate_start_c
            text = (
                f"版本 v{previous.version}→v{params.version}："
                f"热容 {old.capacity_kj_k:.0f}→{params.capacity_kj_k:.0f} kJ/K、"
                f"散热 {old.loss_kw_k:.2f}→{params.loss_kw_k:.2f} kW/K，"
                f"降额起点 {previous.derate_start_c:.1f}→{params.derate_start_c:.1f}℃"
                f"（提前 {earlier:.1f}℃），"
                f"上限 {previous.ceiling_cap_pct:.1f}→{params.ceiling_cap_pct:.1f}%"
            )
        else:
            text = f"与上次一致（{source_label} v{params.version}，起点 {params.derate_start_c:.1f}℃）"
        if toggled:
            text += f"；出口 {temperature_c:.1f}℃ " + ("越过起点，进入降额" if self._active else "回落，退出降额")
        return text

    def current_ceiling_pct(self, temperature_c: float | None = None) -> float:
        """按给定温度（缺省为最近一次判定温度）与当前版本给出出力上限。"""

        params = self._params()
        temp = self._last_temp if temperature_c is None else float(temperature_c)
        return min(_ceiling_interp(self._limits, temp, params.derate_start_c), params.ceiling_cap_pct)

    def last_verdict(self) -> AdaptiveVerdict | None:
        """返回最近一次判定。"""

        return self._last_verdict

    def take_pending_estimate(self) -> ThermalEstimate | None:
        """取走待迁移的估计（没有时返回 ``None``）。"""

        estimate = self._pending
        self._pending = None
        return estimate

    def status(self) -> dict:
        """汇总自算保护状态，供控制台与状态快照读取。"""

        current = self._store.current()
        calibration = self._store.calibration()
        estimator = self._estimator
        return {
            "enabled": self._adaptive.enabled,
            "zone_sources": dict(self._adaptive.zone_sources),
            "current": current.as_dict(),
            "calibration": calibration.as_dict(),
            "versions": [version.as_dict() for version in self._store.versions()],
            "migrations": current.version,
            "rejections": self._store.rejections(),
            "last_rejection": self._store.last_rejection(),
            "estimator": {
                "window": self._adaptive.window,
                "samples": 0 if estimator is None else estimator.samples(),
                "fed": 0 if estimator is None else estimator.fed(),
                "pending": self._pending is not None,
            },
            "zones": {
                "panel": {
                    "source": self.zone_source("panel"),
                    "version": self._params().version,
                    "derate_start_c": round(self._params().derate_start_c, 3),
                    "ceiling_cap_pct": round(self._params().ceiling_cap_pct, 3),
                },
                "trip": {
                    "source": self.zone_source("trip"),
                    "temp_max_c": self._limits.temp_max_c,
                },
            },
            "ceiling_pct": round(self.current_ceiling_pct(), 3),
        }


def replay_verdicts(store: ParameterStore, records: Iterable) -> dict:
    """按判定记录里的版本号重算降额边界，逐字段核对与当时记录是否一致。

    记录必须来自审计流水（``adaptive_verdict`` 动作）；引用的版本在版本库
    里不存在、或重算结果与记录不符，都计入不一致。
    """

    checked = 0
    mismatches: list[dict] = []
    for record in records:
        payload = dict(record.payload)
        try:
            version_no = int(payload.get("version"))
        except (TypeError, ValueError):
            mismatches.append({"ts": record.ts, "version": None, "reason": "missing_version_field"})
            continue
        version = store.get(version_no)
        if version is None:
            mismatches.append({"ts": record.ts, "version": version_no, "reason": "version_not_in_store"})
            continue
        checked += 1
        start, cap = store.derive(version)
        problems: list[str] = []
        if abs(start - float(payload.get("derate_start_c", math.nan))) > REPLAY_TOLERANCE:
            problems.append("derate_start_c")
        if abs(cap - float(payload.get("ceiling_cap_pct", math.nan))) > REPLAY_TOLERANCE:
            problems.append("ceiling_cap_pct")
        ceiling = min(
            _ceiling_interp(store.limits, float(payload.get("temperature_c", math.nan)), start),
            cap,
        )
        if abs(ceiling - float(payload.get("ceiling_pct", math.nan))) > REPLAY_TOLERANCE:
            problems.append("ceiling_pct")
        if problems:
            mismatches.append(
                {
                    "ts": record.ts,
                    "version": version_no,
                    "fields": problems,
                    "reason": "recompute_diverged",
                }
            )
    return {"checked": checked, "consistent": not mismatches, "mismatches": mismatches}
