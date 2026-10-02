"""按当时版本解释保护判定，并核对实时、回放、审计三处的一致性。

实时保护落下的每条判定记录都自带：引擎版本、参数 schema/revision、部位
策略指纹、以及当时的全部输入（温度、功率、流量、参数值、是否在降额中）。
回放时必须**按记录里的版本解释**：

* 引擎版本对得上，才用当前引擎的纯函数复算；
* 对不上（代码里没有该版本的解释器）：标记 ``unknown_engine``，禁止用
  新引擎冒充旧判定；
* 参数 schema 比当前代码新：同样拒绝猜测；
* 策略指纹不同：标记 ``policy_drift``；
* 复算结果（active/ceiling/起点）与落盘值对不上：标记 ``replay_mismatch``；
* revision 链条必须与 ``self_params_committed`` 事件单调一致，否则
  标记 ``revision_chain_broken``。

任何一项不通过，整份核对结论都不可信——调用方（控制台/审计）应按异常
处理，而不是拿一个"看起来差不多"的结果继续运行。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import ReceiverLimits
from . import policy as part_policy
from .protection import (
    ENGINE_VERSION,
    ceiling_at,
    enter_threshold,
)
from .self_params import SCHEMA_VERSION, ParamSet

DERATE_ENTERED = "derate_entered"
DERATE_CLEARED = "derate_cleared"
PARAMS_COMMITTED = "self_params_committed"
PARAMS_RECALIBRATED = "self_params_recalibrated"
PROTECTION_DECISIONS = (DERATE_ENTERED, DERATE_CLEARED)

CEILING_TOLERANCE = 0.05
THRESHOLD_TOLERANCE = 0.05


def params_from_record(record_payload: dict) -> ParamSet:
    """从落盘的 effective/committed 字段还原参数（审计载荷兼容两层结构）。"""

    eff = record_payload.get("effective") or record_payload.get("committed") or {}
    return ParamSet(
        capacity_kj_per_k=float(eff["capacity_kj_per_k"]),
        loss_kw_per_k=float(eff["loss_kw_per_k"]),
        flow_gain=float(eff["flow_gain"]),
        raw_capacity_kj_per_k=float(eff.get("raw_capacity_kj_per_k", eff["capacity_kj_per_k"])),
        raw_loss_kw_per_k=float(eff.get("raw_loss_kw_per_k", eff["loss_kw_per_k"])),
        raw_flow_gain=float(eff.get("raw_flow_gain", eff["flow_gain"])),
        fit_ok=bool(eff.get("fit_ok", False)),
        samples=int(eff.get("samples", 0)),
        quality=float(eff.get("quality", 0.0)),
    )


def _stamp_and_params(record: dict) -> tuple[dict, dict]:
    """从审计记录取出判定戳与其生效参数。

    实时落审计时外层同时带了 ``effective``（当时的生效参数快照）与
    ``decision``（判定戳）；直接构造的回放记录也可能把两者平铺。
    """

    payload = record.get("payload", record)
    stamp = dict(payload.get("decision", payload))
    stamp.setdefault("action", record.get("action", ""))
    if "effective" in payload:
        stamp.setdefault("effective", payload["effective"])
    if "committed" in payload and "effective" not in stamp:
        stamp.setdefault("committed", payload["committed"])
    if "derate_lead_s" in payload and "derate_lead_s" not in stamp:
        stamp["derate_lead_s"] = payload["derate_lead_s"]
    return stamp, payload


def interpret_decision(payload: dict, limits: ReceiverLimits, active_before: bool) -> dict:
    """按记录自带版本解释一条降额判定（纯函数，回放与实时共用口径）。

    返回复算出的 active/ceiling/起点。引擎版本未知时原样抛出
    :class:`KeyError`，由核对器记为 ``unknown_engine``，不做任何猜测。
    """

    engine = int(payload["engine_version"])
    if engine != ENGINE_VERSION:
        raise KeyError(f"unknown_engine={engine}")
    schema = int(payload.get("schema", SCHEMA_VERSION))
    if schema > SCHEMA_VERSION:
        raise KeyError(f"unknown_schema={schema}")

    params = params_from_record(payload)
    temp = float(payload["temperature_c"])
    rate = float(payload["rate_c_s"])
    lead = float(payload.get("derate_lead_s", 0.0))
    threshold = enter_threshold(limits, rate, lead)
    self_ceiling = ceiling_at(temp, threshold, limits)
    cal_ceiling = ceiling_at(temp, limits.temp_derate_c, limits)
    ceiling = min(self_ceiling, cal_ceiling)

    if payload.get("action") == DERATE_ENTERED:
        active = True
    elif payload.get("action") == DERATE_CLEARED:
        active = False
    else:
        active = active_before
    if not active:
        ceiling = 100.0
    return {
        "active": active,
        "ceiling_pct": ceiling,
        "enter_threshold_c": threshold,
        "engine_version": engine,
    }


@dataclass(frozen=True)
class ReplayVerdict:
    """一次回放核对结论。"""

    checked: int
    ok: bool
    issues: list[dict]
    revisions: list[int]

    def as_dict(self) -> dict:
        return {
            "checked": self.checked,
            "ok": self.ok,
            "issues": self.issues,
            "revisions": self.revisions,
            "engine_version": ENGINE_VERSION,
            "schema": SCHEMA_VERSION,
            "policy_fingerprint": part_policy.POLICY_FINGERPRINT,
        }


def verify_protection_records(
    audit_records: list[dict],
    limits: ReceiverLimits,
    *,
    expected_policy_fingerprint: str | None = None,
) -> ReplayVerdict:
    """从审计流水回放全部降额判定并核对一致性。

    ``audit_records`` 为审计记录字典（含 ``action`` 与 ``payload``）。
    实时侧、跨重启回放侧都可以把各自的流水喂进来；结论必须一致。
    """

    issues: list[dict] = []
    revisions: list[int] = []
    active_before = False
    checked = 0
    fingerprint = expected_policy_fingerprint or part_policy.POLICY_FINGERPRINT

    for record in audit_records:
        action = record.get("action", "")
        payload = record.get("payload", {})
        if action == PARAMS_COMMITTED or action == PARAMS_RECALIBRATED:
            rev = int(payload.get("revision", -1))
            revisions.append(rev)
            if revisions.count(rev) > 1 and action == PARAMS_COMMITTED:
                # 同一 revision 重复出现允许（重启恢复），但倒序不允许。
                pass
            continue
        if action not in PROTECTION_DECISIONS:
            continue
        checked += 1
        stamp, payload_top = _stamp_and_params(record)

        if int(stamp.get("engine_version", ENGINE_VERSION)) != ENGINE_VERSION:
            issues.append({"ts": record.get("ts"), "issue": "unknown_engine",
                           "recorded": stamp.get("engine_version"), "current": ENGINE_VERSION})
            continue
        if int(stamp.get("schema", SCHEMA_VERSION)) > SCHEMA_VERSION:
            issues.append({"ts": record.get("ts"), "issue": "unknown_schema",
                           "recorded": stamp.get("schema"), "current": SCHEMA_VERSION})
            continue
        if stamp.get("policy_fingerprint") != fingerprint:
            issues.append({"ts": record.get("ts"), "issue": "policy_drift",
                           "recorded": stamp.get("policy_fingerprint"), "current": fingerprint})
        try:
            replay_payload = dict(stamp)
            replay_payload["action"] = action
            recomputed = interpret_decision(replay_payload, limits, active_before)
        except KeyError as error:
            issues.append({"ts": record.get("ts"), "issue": str(error).strip("'")})
            continue

        recorded_active = bool(stamp.get("active", action == DERATE_ENTERED))
        if recomputed["active"] != recorded_active:
            issues.append({
                "ts": record.get("ts"), "issue": "replay_mismatch",
                "field": "active", "recorded": recorded_active,
                "replayed": recomputed["active"],
            })
        recorded_ceiling = float(stamp.get("ceiling_pct", 100.0))
        if abs(recomputed["ceiling_pct"] - recorded_ceiling) > CEILING_TOLERANCE:
            issues.append({
                "ts": record.get("ts"), "issue": "replay_mismatch",
                "field": "ceiling_pct", "recorded": recorded_ceiling,
                "replayed": round(recomputed["ceiling_pct"], 3),
            })
        recorded_threshold = float(stamp.get("enter_threshold_c", limits.temp_derate_c))
        if abs(recomputed["enter_threshold_c"] - recorded_threshold) > THRESHOLD_TOLERANCE:
            issues.append({
                "ts": record.get("ts"), "issue": "replay_mismatch",
                "field": "enter_threshold_c", "recorded": recorded_threshold,
                "replayed": round(recomputed["enter_threshold_c"], 3),
            })
        active_before = recorded_active

    # revision 必须随时间单调（允许重启后同值重复，不允许倒退）。
    unique_revisions: list[int] = []
    for rev in revisions:
        if not unique_revisions or rev != unique_revisions[-1]:
            unique_revisions.append(rev)
    for earlier, later in zip(unique_revisions, unique_revisions[1:]):
        if later < earlier:
            issues.append({"issue": "revision_chain_broken", "from": earlier, "to": later})

    return ReplayVerdict(
        checked=checked,
        ok=not issues,
        issues=issues,
        revisions=unique_revisions,
    )


__all__ = [
    "DERATE_ENTERED",
    "DERATE_CLEARED",
    "PARAMS_COMMITTED",
    "PARAMS_RECALIBRATED",
    "interpret_decision",
    "verify_protection_records",
    "ReplayVerdict",
    "params_from_record",
]
