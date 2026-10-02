"""版本化的吸热器降额保护模型。

保护引擎做四件事，全部落在同一个版本号（:data:`ENGINE_VERSION`）下：

1. 每个受热步把实际功率、指示流量、原始温升喂给在线辨识器，拿到自算参数；
2. 自算值按"只往保守走"的单向棘轮收紧（热容只降、散热只升、流量增益只降），
   收紧即产生新 revision 并落审计；放松只能走检修后重新标定；
3. 用生效参数外推当前工况下的温升速率，反推"要在 ``derate_lead_s`` 秒后
   仍不越过标定降额线，现在该在什么温度开始降额"，起点只能提前不能推后；
   出力上限曲线取自算曲线与标定曲线中更保守的一条；
4. 每次进入降额都给出结构化依据：相对标定提前多少、相对上一次同工况事件
   早/晚多少，归因到热容、散热、流量增益三个因子。

回放见 :mod:`replay`：同一份纯函数、同一版本、同一参数 revision 必须把
历史判定复算成一致结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..config import ReceiverLimits
from .derate import HYSTERESIS_C
from .estimator import SALT_CP_KJ_KG_K, Estimate, ThermalEstimator
from . import policy as part_policy
from .policy import CAL_ONLY, PANEL, POLICY_FINGERPRINT, SELF_PESSIMUM
from .self_params import (
    COMMIT_DEADBAND,
    CalibratedParams,
    ParamSet,
    SelfParamState,
    SelfParamError,
    initial_state,
    is_more_conservative,
    pessimistic,
    raw_plausible,
)

# 保护引擎版本。判定语义（起点公式、上限合成、回差、来源仲裁）任何改动
# 都必须升版；回放时按记录里的版本解释，不能用新引擎重算旧判定。
ENGINE_VERSION = 1

# 自算值相对标定收紧到这个比例以上，来源才标注为 self。
SELF_SOURCE_DEADBAND = 0.002

# 原始估计的保守向平滑系数：只在"更保守"方向并入，避免噪声来回刷 revision。
RAW_FILTER_ALPHA = 0.05

# 收紧必须连续这么多步保持有效且越过死区才提交，防止单步坏数据触发棘轮。
COMMIT_CONFIRM_STEPS = 30


@dataclass(frozen=True)
class Decision:
    """一次降额判定结果，字段即回放所需的全部输入与结论。"""

    active: bool
    ceiling_pct: float
    temperature_c: float
    enter_threshold_c: float
    rate_c_s: float
    source: str
    changed: str
    revision: int
    schema: int
    engine_version: int
    policy_fingerprint: str
    part: str
    explanation: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "active": self.active,
            "ceiling_pct": round(self.ceiling_pct, 3),
            "temperature_c": round(self.temperature_c, 3),
            "enter_threshold_c": round(self.enter_threshold_c, 3),
            "rate_c_s": round(self.rate_c_s, 5),
            "source": self.source,
            "changed": self.changed,
            "revision": self.revision,
            "schema": self.schema,
            "engine_version": self.engine_version,
            "policy_fingerprint": self.policy_fingerprint,
            "part": self.part,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class Observation:
    """一步受热后的自算结果：原始估计、是否收紧、保守生效值。"""

    estimate: Estimate
    committed: ParamSet
    revision_before: int
    committed_now: bool
    reason: str

    def as_dict(self) -> dict:
        return {
            "raw": self.estimate.raw.as_dict(),
            "committed": self.committed.as_dict(),
            "revision_before": self.revision_before,
            "committed_now": self.committed_now,
            "reason": self.reason,
            "residual_c_s": round(self.estimate.residual_c_s, 5),
            "residual_ratio": round(self.estimate.residual_ratio, 6),
            "capacity_identifiable": self.estimate.capacity_identifiable,
            "cooling_identifiable": self.estimate.cooling_identifiable,
            "eigenvalues": [round(value, 8) for value in self.estimate.eigenvalues],
        }


def effective_for_part(state: SelfParamState, part: str) -> ParamSet:
    """按部位冻结策略给出保护实际采用的参数。

    受热管屏：自算悲观值已在提交时按标定夹过，直接采用；
    其余部位：只取标定，自算值永不参与保护。
    """

    source = part_policy.policy_for(part).source
    cal = state.calibrated
    if source == CAL_ONLY:
        return ParamSet(
            capacity_kj_per_k=cal.capacity_kj_per_k,
            loss_kw_per_k=cal.loss_kw_per_k,
            flow_gain=cal.flow_gain,
            raw_capacity_kj_per_k=state.committed.raw_capacity_kj_per_k,
            raw_loss_kw_per_k=state.committed.raw_loss_kw_per_k,
            raw_flow_gain=state.committed.raw_flow_gain,
            fit_ok=state.committed.fit_ok,
            samples=state.committed.samples,
            quality=state.committed.quality,
        )
    return state.committed


def projected_rate_c_s(params: ParamSet, power_kw: float, flow_indicated_kg_s: float,
                       temperature_c: float, inlet_c: float, ambient_c: float,
                       cp_kj_kg_k: float = SALT_CP_KJ_KG_K) -> float:
    """按生效参数外推当前工况下的净温升速率（℃/s），不为负取 0。"""

    carried = params.flow_gain * cp_kj_kg_k * max(0.0, flow_indicated_kg_s) * (temperature_c - inlet_c)
    lost = params.loss_kw_per_k * (temperature_c - ambient_c)
    return max(0.0, (float(power_kw) - carried - lost) / params.capacity_kj_per_k)


def enter_threshold(limits: ReceiverLimits, rate_c_s: float, lead_s: float) -> float:
    """反推降额起点：给 ``lead_s`` 秒提前量，起点只能比标定线更早。"""

    projected = limits.temp_derate_c - max(0.0, rate_c_s) * max(0.0, lead_s)
    return max(limits.temp_target_c, min(limits.temp_derate_c, projected))


def threshold_fixed_point(
    params: ParamSet,
    power_kw: float,
    flow_indicated_kg_s: float,
    limits: ReceiverLimits,
    inlet_c: float,
    ambient_c: float,
    lead_s: float,
) -> tuple[float, float]:
    """求起点阈值与速率的不动点 ``T* = clamp(T_derate − rate(T*)·lead)``。

    速率随温度升高而变小（带走热量在增大），迭代单调收敛；受热路径在
    阈值温度点上判定，必须用阈值自身温度下的速率，否则起点与速率会
    自相矛盾。投影被钳到目标温度时，速率也要在钳点处重算，保证记录里
    的"提前秒数 × 速率"正好等于温度提前量。返回 ``(阈值, 阈值处速率)``。
    """

    threshold = limits.temp_derate_c
    for _ in range(20):
        rate = projected_rate_c_s(params, power_kw, flow_indicated_kg_s, threshold, inlet_c, ambient_c)
        next_threshold = enter_threshold(limits, rate, lead_s)
        if abs(next_threshold - threshold) < 1e-6:
            break
        threshold = next_threshold
    # 收敛后（含被钳到目标温度的情形）在最终阈值点重算一次速率。
    rate = projected_rate_c_s(params, power_kw, flow_indicated_kg_s, threshold, inlet_c, ambient_c)
    return enter_threshold(limits, rate, lead_s), rate


def ceiling_at(temperature_c: float, threshold_c: float, limits: ReceiverLimits) -> float:
    """自算锚点下的出力上限曲线（100% → 0%）。"""

    if temperature_c <= threshold_c:
        return 100.0
    span = limits.temp_max_c - threshold_c
    if span <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * (limits.temp_max_c - temperature_c) / span))


def calibration_ceiling_at(temperature_c: float, limits: ReceiverLimits) -> float:
    """标定锚点下的出力上限曲线（与旧 DerateController 同口径）。"""

    return ceiling_at(temperature_c, limits.temp_derate_c, limits)


def _source_label(state: SelfParamState) -> str:
    """当前生效值相对标定是否已实质收紧。"""

    cal, com = state.calibrated, state.committed
    tighter = (
        com.capacity_kj_per_k < cal.capacity_kj_per_k * (1.0 - SELF_SOURCE_DEADBAND)
        or com.loss_kw_per_k > cal.loss_kw_per_k * (1.0 + SELF_SOURCE_DEADBAND)
        or com.flow_gain < cal.flow_gain * (1.0 - SELF_SOURCE_DEADBAND)
    )
    return SELF_PESSIMUM if tighter else CAL_ONLY


def _consistent_with_pending(pending: ParamSet, candidate: ParamSet) -> bool:
    """候选值是否仍落在待确认值附近（容差半个提交死区），用于连续确认。"""

    tol = COMMIT_DEADBAND * 0.5
    return (
        abs(candidate.capacity_kj_per_k - pending.capacity_kj_per_k) / pending.capacity_kj_per_k <= tol
        and abs(candidate.loss_kw_per_k - pending.loss_kw_per_k) / max(1.0, pending.loss_kw_per_k) <= tol
        and abs(candidate.flow_gain - pending.flow_gain) / pending.flow_gain <= tol
    )


class ProtectionModel:
    """按部位、按版本解释自算/标定参数的降额保护模型。"""

    def __init__(
        self,
        limits: ReceiverLimits,
        calibrated: CalibratedParams,
        *,
        part: str = PANEL,
        derate_lead_s: float = 20.0,
        inlet_c: float = 290.0,
        ambient_c: float | None = None,
        state: SelfParamState | None = None,
    ) -> None:
        if part not in part_policy.PARTS:
            raise SelfParamError(f"未登记的吸热器部位: {part}", component="receiver")
        self._limits = limits
        self._part = part
        self._lead_s = max(0.0, float(derate_lead_s))
        self._inlet_c = float(inlet_c)
        self._ambient_c = float(limits.ambient_c if ambient_c is None else ambient_c)
        self._state = state or initial_state(calibrated)
        self._estimator = ThermalEstimator(self._state.calibrated, inlet_c=self._inlet_c, ambient_c=self._ambient_c)
        self._active = False
        self._entries = 0
        self._clears = 0
        self._evaluations = 0
        self._min_ceiling = 100.0
        self._last_temp = float(limits.temp_target_c)
        self._last_rate = 0.0
        self._entry_history: list[dict] = []
        self._last_decision: Decision | None = None
        self._last_power = 0.0
        self._last_flow = 0.0
        self._last_now = 0.0
        self._operator = ""
        self._filtered: ParamSet | None = None
        self._pending: ParamSet | None = None
        self._pending_streak = 0

    # ---- 状态与版本 ---------------------------------------------------

    @property
    def limits(self) -> ReceiverLimits:
        return self._limits

    @property
    def part(self) -> str:
        return self._part

    @property
    def lead_s(self) -> float:
        return self._lead_s

    @property
    def state(self) -> SelfParamState:
        return self._state

    @property
    def estimator(self) -> ThermalEstimator:
        return self._estimator

    @property
    def engine_version(self) -> int:
        return ENGINE_VERSION

    @property
    def policy_fingerprint(self) -> str:
        return POLICY_FINGERPRINT

    def source(self) -> str:
        """当前部位保护实际采用的来源策略（已冻结，不受运行时影响）。"""

        return part_policy.policy_for(self._part).source

    def active(self) -> bool:
        return self._active

    def entries(self) -> int:
        return self._entries

    def clears(self) -> int:
        return self._clears

    def evaluation_count(self) -> int:
        return self._evaluations

    def min_ceiling_pct(self) -> float:
        return self._min_ceiling

    def last_temperature_c(self) -> float:
        return self._last_temp

    def last_rate_c_s(self) -> float:
        return self._last_rate

    def entry_history(self) -> list[dict]:
        return list(self._entry_history)

    def last_decision(self) -> Decision | None:
        return self._last_decision

    def effective_params(self) -> ParamSet:
        """返回当前部位保护采用的参数（面板=自算悲观值，其余=标定）。"""

        return effective_for_part(self._state, self._part)

    # ---- 自算与单向棘轮 -----------------------------------------------

    def observe(self, power_kw: float, flow_indicated_kg_s: float, raw_temp_c: float, dt: float,
                now: float = 0.0) -> Observation:
        """喂一步实测信号；只有面板部位才允许自算收紧生效值。"""

        estimate = self._estimator.observe(power_kw, flow_indicated_kg_s, raw_temp_c, dt)
        revision_before = self._state.revision
        can_self = part_policy.uses_self_params(self._part)
        committed_now = False
        reason = "pending"
        candidate = self._state.committed

        if not can_self:
            reason = f"part_{self._part}_cal_only"
        elif not estimate.started:
            reason = "warming_up"
        elif not (estimate.capacity_identifiable or estimate.cooling_identifiable):
            reason = "fit_rejected"
        elif not raw_plausible(estimate.raw):
            reason = "outside_plausible_envelope"
            self._pending = None
            self._pending_streak = 0
        else:
            # 只对"可辨识且更保守"的维度取值；不可分维度钉死在标定值上，
            # 即使拟合残差很小也不允许沿共线方向瞎猜。
            cal = self._state.calibrated
            ident = estimate.raw
            dimension_gated = ParamSet(
                capacity_kj_per_k=ident.capacity_kj_per_k if estimate.capacity_identifiable else cal.capacity_kj_per_k,
                loss_kw_per_k=ident.loss_kw_per_k if estimate.cooling_identifiable else cal.loss_kw_per_k,
                flow_gain=ident.flow_gain if estimate.cooling_identifiable else cal.flow_gain,
                raw_capacity_kj_per_k=estimate.raw.raw_capacity_kj_per_k,
                raw_loss_kw_per_k=estimate.raw.raw_loss_kw_per_k,
                raw_flow_gain=estimate.raw.raw_flow_gain,
                fit_ok=estimate.raw.fit_ok,
                samples=estimate.raw.samples,
                quality=estimate.raw.quality,
            )
            candidate = self._filter_raw(pessimistic(cal, dimension_gated))
            if is_more_conservative(self._state.committed, candidate):
                # 只朝保守方向累积待确认值；连续 N 步稳定才提交，中途若
                # 候选在死区内抖动视为同一值，朝宽松漂则清零重来。
                if self._pending is not None and not _consistent_with_pending(self._pending, candidate):
                    self._pending = candidate
                    self._pending_streak = 1
                else:
                    if self._pending is None:
                        self._pending = candidate
                    self._pending_streak += 1
                reason = "tightening_pending"
                if self._pending_streak >= COMMIT_CONFIRM_STEPS and is_more_conservative(
                    self._state.committed, self._pending
                ):
                    confirmed = replace(
                        self._pending,
                        raw_capacity_kj_per_k=candidate.raw_capacity_kj_per_k,
                        raw_loss_kw_per_k=candidate.raw_loss_kw_per_k,
                        raw_flow_gain=candidate.raw_flow_gain,
                        fit_ok=True,
                        samples=candidate.samples,
                        quality=candidate.quality,
                    )
                    self._state = replace(
                        self._state,
                        committed=confirmed,
                        revision=self._state.revision + 1,
                        updated_ts=float(now),
                    )
                    committed_now = True
                    reason = "ratchet_tightened"
                    self._pending = None
                    self._pending_streak = 0
            else:
                self._pending = None
                self._pending_streak = 0
                # 拟合有效但不更保守：记录原始值供展示，生效值不动。
                self._state = replace(
                    self._state,
                    committed=replace(
                        self._state.committed,
                        raw_capacity_kj_per_k=candidate.raw_capacity_kj_per_k,
                        raw_loss_kw_per_k=candidate.raw_loss_kw_per_k,
                        raw_flow_gain=candidate.raw_flow_gain,
                        fit_ok=True,
                        samples=candidate.samples,
                        quality=candidate.quality,
                    ),
                    updated_ts=float(now),
                )
                reason = "not_more_conservative"
        return Observation(estimate, self._state.committed, revision_before, committed_now, reason)

    def _filter_raw(self, candidate: ParamSet) -> ParamSet:
        """对候选生效值做保守向平滑。

        只朝更保守方向并入新观测（热容取小、散热取大、增益取小），因此
        滤波只会延迟收紧，永远不会把已收紧的值放松；原始观测仍原样留在
        raw_* 字段供展示。
        """

        a = RAW_FILTER_ALPHA
        if self._filtered is None:
            self._filtered = candidate
            return candidate
        previous = self._filtered
        smoothed = ParamSet(
            capacity_kj_per_k=min(previous.capacity_kj_per_k,
                                  a * candidate.capacity_kj_per_k + (1.0 - a) * previous.capacity_kj_per_k),
            loss_kw_per_k=max(previous.loss_kw_per_k,
                              a * candidate.loss_kw_per_k + (1.0 - a) * previous.loss_kw_per_k),
            flow_gain=min(previous.flow_gain,
                          a * candidate.flow_gain + (1.0 - a) * previous.flow_gain),
            raw_capacity_kj_per_k=candidate.raw_capacity_kj_per_k,
            raw_loss_kw_per_k=candidate.raw_loss_kw_per_k,
            raw_flow_gain=candidate.raw_flow_gain,
            fit_ok=candidate.fit_ok,
            samples=candidate.samples,
            quality=candidate.quality,
        )
        self._filtered = smoothed
        return smoothed

    def recalibrate(self, capacity_kj_per_k: float, loss_kw_per_k: float, flow_gain: float,
                    now: float, operator: str = "maintenance") -> SelfParamState:
        """检修后重新标定：棘轮唯一的放松入口，仍受物理包络约束并升 revision。

        新标定是新的宽松边界，生效值重置到新标定（允许放松），但这次放松
        必须带操作人落审计；旧的已收紧值不会在无人知晓的情况下被覆盖。
        """

        calibrated = CalibratedParams(float(capacity_kj_per_k), float(loss_kw_per_k), float(flow_gain))
        if not (1_000.0 < calibrated.capacity_kj_per_k <= 100_000.0):
            raise SelfParamError("标定热容超出合理范围", component="receiver")
        if not (0.0 <= calibrated.loss_kw_per_k <= 200.0):
            raise SelfParamError("标定散热系数超出合理范围", component="receiver")
        if not (0.5 <= calibrated.flow_gain <= 1.5):
            raise SelfParamError("标定流量增益超出合理范围", component="receiver")
        committed = ParamSet(
            capacity_kj_per_k=calibrated.capacity_kj_per_k,
            loss_kw_per_k=calibrated.loss_kw_per_k,
            flow_gain=calibrated.flow_gain,
            raw_capacity_kj_per_k=calibrated.capacity_kj_per_k,
            raw_loss_kw_per_k=calibrated.loss_kw_per_k,
            raw_flow_gain=calibrated.flow_gain,
            fit_ok=False,
            samples=0,
            quality=0.0,
        )
        self._state = SelfParamState(
            calibrated=calibrated,
            committed=committed,
            schema=self._state.schema,
            revision=self._state.revision + 1,
            updated_ts=float(now),
        )
        self._estimator = ThermalEstimator(calibrated, inlet_c=self._inlet_c, ambient_c=self._ambient_c)
        self._last_rate = 0.0
        self._filtered = None
        self._pending = None
        self._pending_streak = 0
        self._operator = str(operator)
        return self._state

    def restore_state(self, state: SelfParamState) -> None:
        """跨重启恢复自算状态（revision 与生效值原样带回）。"""

        self._state = state
        self._estimator = ThermalEstimator(state.calibrated, inlet_c=self._inlet_c, ambient_c=self._ambient_c)
        self._filtered = None

    # ---- 判定 ---------------------------------------------------------

    def evaluate(
        self,
        temperature_c: float,
        power_kw: float,
        flow_indicated_kg_s: float,
        dt: float = 1.0,
        rate_c_s: float | None = None,
    ) -> Decision:
        """按当前部位来源与生效参数做一次降额判定（纯口径，可回放）。

        ``rate_c_s`` 可显式给定（解释/回放时用记录值）；实时路径默认按
        当前工况外推。
        """

        if rate_c_s is None:
            params = effective_for_part(self._state, self._part)
            rate_c_s = projected_rate_c_s(
                params, power_kw, flow_indicated_kg_s, temperature_c, self._inlet_c, self._ambient_c
            )
        return self._evaluate(
            temperature_c,
            effective_for_part(self._state, self._part),
            power_kw,
            flow_indicated_kg_s,
            dt,
            rate_c_s=rate_c_s,
            source_hint=_source_label(self._state) if part_policy.uses_self_params(self._part) else CAL_ONLY,
        )

    def _evaluate(
        self,
        temperature_c: float,
        params: ParamSet,
        power_kw: float,
        flow_indicated_kg_s: float,
        dt: float,
        *,
        source_hint: str,
        rate_c_s: float | None = None,
    ) -> Decision:
        self._evaluations += 1
        self._last_temp = float(temperature_c)
        if rate_c_s is None:
            # 普通查询：求阈值不动点，保证起点与阈值温度下的速率自洽。
            threshold, rate_c_s = threshold_fixed_point(
                params, power_kw, flow_indicated_kg_s, self._limits,
                self._inlet_c, self._ambient_c, self._lead_s,
            )
        else:
            threshold = enter_threshold(self._limits, float(rate_c_s), self._lead_s)
        window_rate = float(rate_c_s)
        self._last_rate = window_rate

        self_ceiling = ceiling_at(temperature_c, threshold, self._limits)
        cal_ceiling = calibration_ceiling_at(temperature_c, self._limits)
        # 自算曲线永远不得比标定曲线宽松。
        ceiling = min(self_ceiling, cal_ceiling)
        source = source_hint if ceiling < cal_ceiling - 1e-9 else CAL_ONLY
        self._min_ceiling = min(self._min_ceiling, ceiling)

        changed = ""
        explanation: dict = {}
        if not self._active and temperature_c >= threshold:
            self._active = True
            self._entries += 1
            changed = "entered"
            explanation = self._explain_entry(temperature_c, window_rate, threshold, params)
        elif self._active and temperature_c < threshold - HYSTERESIS_C:
            self._active = False
            self._clears += 1
            changed = "cleared"
            explanation = self._explain_clear(temperature_c, threshold)
        ceiling = ceiling if self._active else 100.0

        decision = Decision(
            active=self._active,
            ceiling_pct=ceiling,
            temperature_c=float(temperature_c),
            enter_threshold_c=threshold,
            rate_c_s=window_rate,
            source=source,
            changed=changed,
            revision=self._state.revision,
            schema=self._state.schema,
            engine_version=ENGINE_VERSION,
            policy_fingerprint=POLICY_FINGERPRINT,
            part=self._part,
            explanation=explanation,
        )
        self._last_decision = decision
        return decision

    def _factor_deltas(self, params: ParamSet, baseline: CalibratedParams) -> list[dict]:
        return [
            {
                "factor": "capacity_kj_per_k",
                "self": round(params.capacity_kj_per_k, 2),
                "calibration": round(baseline.capacity_kj_per_k, 2),
                "ratio": round(params.capacity_kj_per_k / baseline.capacity_kj_per_k, 4),
                "effect": "smaller_capacity_heats_faster",
            },
            {
                "factor": "loss_kw_per_k",
                "self": round(params.loss_kw_per_k, 2),
                "calibration": round(baseline.loss_kw_per_k, 2),
                "ratio": round(params.loss_kw_per_k / baseline.loss_kw_per_k, 4),
                "effect": "larger_loss_tightens_balance",
            },
            {
                "factor": "flow_gain",
                "self": round(params.flow_gain, 4),
                "calibration": round(baseline.flow_gain, 4),
                "ratio": round(params.flow_gain / baseline.flow_gain, 4),
                "effect": "meter_overreads_real_cooling",
            },
        ]

    def _explain_entry(self, temperature_c: float, rate_c_s: float, threshold_c: float,
                       params: ParamSet) -> dict:
        """给出本次进入降额"凭什么早/晚"的结构化依据。"""

        baseline = self._state.calibrated
        cal_rate = projected_rate_c_s(
            ParamSet(
                capacity_kj_per_k=baseline.capacity_kj_per_k,
                loss_kw_per_k=baseline.loss_kw_per_k,
                flow_gain=baseline.flow_gain,
                raw_capacity_kj_per_k=baseline.capacity_kj_per_k,
                raw_loss_kw_per_k=baseline.loss_kw_per_k,
                raw_flow_gain=baseline.flow_gain,
                fit_ok=False,
                samples=0,
                quality=0.0,
            ),
            self._last_power,
            self._last_flow,
            temperature_c,
            self._inlet_c,
            self._ambient_c,
        )
        # 本次相对纯标定：温度提前量（℃）与时间提前量（s），只可能 ≥0。
        margin_c = max(0.0, self._limits.temp_derate_c - threshold_c)
        earlier_vs_cal_s = margin_c / rate_c_s if rate_c_s > 1e-9 else self._lead_s
        factors = self._factor_deltas(params, baseline)

        previous = self._entry_history[-1] if self._entry_history else None
        vs_previous: dict = {}
        if previous is not None:
            # 同口径对比：当前提前量与上次提前量之差，正=更早，负=更晚。
            current_lead = margin_c / rate_c_s if rate_c_s > 1e-9 else self._lead_s
            prior_denom = previous["rate_c_s"] if previous["rate_c_s"] > 1e-9 else 1e-9
            prior_lead = max(0.0, self._limits.temp_derate_c - previous["enter_threshold_c"]) / prior_denom
            delta_s = round(current_lead - prior_lead, 3)
            vs_previous = {
                "prior_event_ts": previous["ts"],
                "prior_revision": previous["revision"],
                "lead_delta_s": delta_s,
                "direction": "earlier" if delta_s > 0.1 else ("later" if delta_s < -0.1 else "same"),
                "revision_changed": self._state.revision != previous["revision"],
            }

        explanation = {
            "basis": "self_pessimum_projected_rate" if margin_c > 1e-9 else "calibration_threshold",
            "temperature_margin_c": round(margin_c, 3),
            "earlier_vs_calibration_s": round(earlier_vs_cal_s, 3),
            "calibration_rate_c_s": round(cal_rate, 5),
            "projected_rate_c_s": round(rate_c_s, 5),
            "derate_lead_s": self._lead_s,
            "revision": self._state.revision,
            "factors": factors,
        }
        if vs_previous:
            explanation["vs_previous"] = vs_previous
        entry = {
            "ts": self._last_now,
            "temperature_c": round(temperature_c, 3),
            "enter_threshold_c": round(threshold_c, 3),
            "rate_c_s": round(rate_c_s, 5),
            "revision": self._state.revision,
            "source": _source_label(self._state),
        }
        self._entry_history.append(entry)
        return explanation

    def _explain_clear(self, temperature_c: float, threshold_c: float) -> dict:
        return {
            "basis": "hysteresis_below_self_threshold",
            "cleared_below_c": round(threshold_c - HYSTERESIS_C, 3),
            "temperature_c": round(temperature_c, 3),
            "revision": self._state.revision,
        }

    def evaluate_with_context(
        self,
        temperature_c: float,
        power_kw: float,
        flow_indicated_kg_s: float,
        dt: float,
        now: float,
        rate_c_s: float | None = None,
    ) -> Decision:
        """带时间戳与上下文的判定（解释"比上次早晚"需要）。"""

        self._last_power = float(power_kw)
        self._last_flow = float(flow_indicated_kg_s)
        self._last_now = float(now)
        return self.evaluate(temperature_c, power_kw, flow_indicated_kg_s, dt, rate_c_s=rate_c_s)

    def status(self) -> dict:
        """控制台展示与审计用的自算/保护状态。"""

        committed = self._state.committed
        return {
            "part": self._part,
            "source_policy": part_policy.policy_for(self._part).source,
            "source_active": _source_label(self._state),
            "engine_version": ENGINE_VERSION,
            "policy_fingerprint": POLICY_FINGERPRINT,
            "schema": self._state.schema,
            "revision": self._state.revision,
            "updated_ts": round(self._state.updated_ts, 3),
            "calibrated": self._state.calibrated.as_dict(),
            "committed": committed.as_dict(),
            "effective": self.effective_params().as_dict(),
            "derate_lead_s": self._lead_s,
            "active": self._active,
            "entries": self._entries,
            "clears": self._clears,
            "evaluations": self._evaluations,
            "min_ceiling_pct": round(self._min_ceiling, 3),
            "last_rate_c_s": round(self._last_rate, 5),
            "entry_history": list(self._entry_history),
        }
