"""吸热器热工参数的自算结果与版本。

降额保护原来只认固定的热容与散热系数：结焦后金属等效热容变化、保温老化后
散热变强、流量计漂移后"指示流量"高估真实冷却能力，同样工况下保护就会
忽早忽晚。自算参数用实际功率、指示流量与实测温升在线辨识这三个量：

* ``capacity_kj_per_k``：等效热容 C（kJ/K）；
* ``loss_kw_per_k``：按温差的散热系数 UA（kW/K）；
* ``flow_gain``：流量计增益 g，真实流量 = 指示流量 × g（漂移后 g<1）。

三条铁律写死在这个模块里：

1. **只往保守走**。自算生效值相对标定只能：C 取小、UA 取大、g 取小——
   即升温更快、散热更小、可信流量更少，保护只会提前、不会推迟。
2. **冲突按部位定**。受热管屏采用"自算悲观值"（self_pessimum），集箱／
   支撑等部位只用标定（cal_only）；策略由 :mod:`policy` 按部位冻结，
   不允许运行时临时切换。
3. **自算带版本，改版即迁移**。参数结构有模式版本（``schema``），自算结果
   本身有单调递增的 ``revision``；schema 变更必须走 :data:`MIGRATIONS`
   里登记的迁移函数，禁止用新代码静默解释旧参数。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from ..errors import PlantError

# 当前自算参数结构版本。新增字段或改变语义必须升版并登记迁移。
SCHEMA_VERSION = 2

# 首版只辨识热容与散热两项（无 flow_gain）。
INITIAL_SCHEMA = 1


class SelfParamError(PlantError):
    """自算参数版本未知或取值非法。"""


@dataclass(frozen=True)
class CalibratedParams:
    """标定值（出厂／检修后给定），是自算结果不可逾越的宽松边界。"""

    capacity_kj_per_k: float
    loss_kw_per_k: float
    flow_gain: float = 1.0

    def as_dict(self) -> dict:
        return {
            "capacity_kj_per_k": round(self.capacity_kj_per_k, 4),
            "loss_kw_per_k": round(self.loss_kw_per_k, 4),
            "flow_gain": round(self.flow_gain, 6),
        }


@dataclass(frozen=True)
class ParamSet:
    """某一时刻的自算参数：标定、生效（保护实际采用）与最近一次原始估计。

    ``committed`` 是已经按"只往保守走"锁定、供保护使用的值；``raw`` 只是
    辨识器给出的原始结果，未经保守裁剪，不能直接驱动保护。
    """

    capacity_kj_per_k: float
    loss_kw_per_k: float
    flow_gain: float
    raw_capacity_kj_per_k: float
    raw_loss_kw_per_k: float
    raw_flow_gain: float
    fit_ok: bool
    samples: int
    quality: float

    def as_dict(self) -> dict:
        return {
            "capacity_kj_per_k": round(self.capacity_kj_per_k, 4),
            "loss_kw_per_k": round(self.loss_kw_per_k, 4),
            "flow_gain": round(self.flow_gain, 6),
            "raw_capacity_kj_per_k": round(self.raw_capacity_kj_per_k, 4),
            "raw_loss_kw_per_k": round(self.raw_loss_kw_per_k, 4),
            "raw_flow_gain": round(self.raw_flow_gain, 6),
            "fit_ok": self.fit_ok,
            "samples": self.samples,
            "quality": round(self.quality, 4),
        }


@dataclass(frozen=True)
class SelfParamState:
    """可持久化的自算参数状态。

    ``schema`` 是结构版本（改版属于迁移）；``revision`` 是生效值的代际，
    每收紧一次加一，只能单调递增——回放时靠它核对参数链条的一致性。
    """

    calibrated: CalibratedParams
    committed: ParamSet
    schema: int = SCHEMA_VERSION
    revision: int = 0
    updated_ts: float = 0.0

    def as_dict(self) -> dict:
        return {
            "schema": self.schema,
            "revision": self.revision,
            "updated_ts": round(self.updated_ts, 3),
            "calibrated": self.calibrated.as_dict(),
            "committed": self.committed.as_dict(),
        }


def initial_state(calibrated: CalibratedParams, now: float = 0.0) -> SelfParamState:
    """构造首版自算状态：没有可信辨识结果时，生效值与标定一致。"""

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
    return SelfParamState(calibrated=calibrated, committed=committed, schema=SCHEMA_VERSION, revision=0, updated_ts=now)


def pessimistic(calibrated: CalibratedParams, raw: ParamSet) -> ParamSet:
    """把原始估计按标定裁剪成"只会更保守"的生效值。

    C：自算比标定小才采纳（升温更快、缓冲更小）；
    UA：自算比标定大才采纳（净吸热更多）；
    g：自算比标定小才采纳（真实冷却能力更低）。
    再做一次物理下限保护，避免辨识异常把保护值钉死。
    """

    capacity = min(raw.capacity_kj_per_k, calibrated.capacity_kj_per_k)
    loss = max(raw.loss_kw_per_k, calibrated.loss_kw_per_k)
    gain = min(raw.flow_gain, calibrated.flow_gain)
    return replace(
        raw,
        capacity_kj_per_k=max(capacity, MIN_CAPACITY_KJ_K),
        loss_kw_per_k=min(loss, MAX_LOSS_KW_K),
        flow_gain=max(gain, MIN_FLOW_GAIN),
    )


# 物理合理性包络：超出即视为辨识异常，绝不允许收紧到这些边界之外。
MIN_CAPACITY_KJ_K = 1_000.0
MAX_LOSS_KW_K = 200.0
MIN_FLOW_GAIN = 0.5
MAX_FLOW_GAIN = 1.5


def raw_plausible(raw: ParamSet) -> bool:
    """原始估计是否落在物理合理包络内。"""

    return (
        MIN_CAPACITY_KJ_K <= raw.capacity_kj_per_k <= 100_000.0
        and 0.0 <= raw.loss_kw_per_k <= MAX_LOSS_KW_K
        and MIN_FLOW_GAIN <= raw.flow_gain <= MAX_FLOW_GAIN
    )


# 单步收紧必须超过这个门槛才推进 revision：小于门槛的差异落在辨识噪声
# 带内，累计漂移仍会逐步越过门槛，但不会每步刷一个版本。
COMMIT_DEADBAND = 0.005


def is_more_conservative(current: ParamSet, candidate: ParamSet) -> bool:
    """候选生效值是否在任一维度上比当前值更保守（且不出现放松）。

    自算是单向棘轮：任何维度的放松都必须拒绝（放松只能走检修后重新标定，
    见 :meth:`SelfParamStore.recalibrate`）。
    """

    c_tighter = candidate.capacity_kj_per_k < current.capacity_kj_per_k * (1.0 - COMMIT_DEADBAND)
    ua_tighter = candidate.loss_kw_per_k > current.loss_kw_per_k * (1.0 + COMMIT_DEADBAND)
    g_tighter = candidate.flow_gain < current.flow_gain * (1.0 - COMMIT_DEADBAND)
    relaxed = (
        candidate.capacity_kj_per_k > current.capacity_kj_per_k * (1.0 + COMMIT_DEADBAND)
        or candidate.loss_kw_per_k < current.loss_kw_per_k * (1.0 - COMMIT_DEADBAND)
        or candidate.flow_gain > current.flow_gain * (1.0 + COMMIT_DEADBAND)
    )
    return (c_tighter or ua_tighter or g_tighter) and not relaxed


# ---- 版本迁移 ---------------------------------------------------------

# 迁移链：每个函数把某个 schema 的落盘文档升级到下一版。
# v1 只有热容／散热两个标量，没有流量计增益；升级时按"无漂移"补 g=1.0。
def _migrate_1_to_2(raw: dict) -> dict:
    migrated = dict(raw)
    migrated["schema"] = 2
    calibrated = dict(migrated.get("calibrated", {}))
    committed = dict(migrated.get("committed", {}))
    calibrated.setdefault("flow_gain", 1.0)
    committed.setdefault("flow_gain", calibrated["flow_gain"])
    committed.setdefault("raw_flow_gain", calibrated["flow_gain"])
    migrated["calibrated"] = calibrated
    migrated["committed"] = committed
    return migrated


MIGRATIONS: dict[int, tuple[int, Callable[[dict], dict]]] = {
    1: (2, _migrate_1_to_2),
}


def migrate_document(raw: dict) -> tuple[dict, list[int]]:
    """把任意已登记的旧版文档迁移到当前 schema，返回新文档与经过的版本。

    遇到未知（含比代码更新的）schema 直接抛错：保护参数宁可拒绝启动、
    退回纯标定，也不允许猜测着解释。
    """

    if not isinstance(raw, dict):
        raise SelfParamError("自算参数文档必须是映射", component="receiver")
    doc = dict(raw)
    applied: list[int] = []
    seen: set[int] = set()
    while int(doc.get("schema", INITIAL_SCHEMA)) != SCHEMA_VERSION:
        schema = int(doc.get("schema", INITIAL_SCHEMA))
        if schema in seen or schema not in MIGRATIONS:
            raise SelfParamError(
                f"未知的自算参数版本 schema={schema}，当前代码只支持到 {SCHEMA_VERSION}",
                component="receiver",
            )
        target, migrator = MIGRATIONS[schema]
        doc = migrator(doc)
        applied.append(target)
        seen.add(schema)
    return doc, applied


def state_from_dict(raw: dict) -> SelfParamState:
    """从（已迁移到当前 schema 的）文档还原自算状态。"""

    doc, _applied = migrate_document(raw)
    cal = dict(doc["calibrated"])
    com = dict(doc["committed"])
    calibrated = CalibratedParams(
        capacity_kj_per_k=float(cal["capacity_kj_per_k"]),
        loss_kw_per_k=float(cal["loss_kw_per_k"]),
        flow_gain=float(cal.get("flow_gain", 1.0)),
    )
    committed = ParamSet(
        capacity_kj_per_k=float(com["capacity_kj_per_k"]),
        loss_kw_per_k=float(com["loss_kw_per_k"]),
        flow_gain=float(com.get("flow_gain", calibrated.flow_gain)),
        raw_capacity_kj_per_k=float(com.get("raw_capacity_kj_per_k", com["capacity_kj_per_k"])),
        raw_loss_kw_per_k=float(com.get("raw_loss_kw_per_k", com["loss_kw_per_k"])),
        raw_flow_gain=float(com.get("raw_flow_gain", calibrated.flow_gain)),
        fit_ok=bool(com.get("fit_ok", False)),
        samples=int(com.get("samples", 0)),
        quality=float(com.get("quality", 0.0)),
    )
    return SelfParamState(
        calibrated=calibrated,
        committed=committed,
        schema=SCHEMA_VERSION,
        revision=int(doc.get("revision", 0)),
        updated_ts=float(doc.get("updated_ts", 0.0)),
    )
