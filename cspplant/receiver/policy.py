"""按部位冻结的保护参数来源策略。

自算结果与标定冲突时听谁的，必须**按部位定死**，不能运行时看哪个顺眼用
哪个：

* 受热管屏（``PANEL``）：热工信号充分、辨识可验证，采用 ``self_pessimum``
  ——自算生效值只在比标定更保守时接管保护，否则沿用标定；
* 集箱、支撑结构（``HEADER`` / ``SUPPORT``）：没有独立的功率／温升观测，
  自算不可辨识，一律 ``cal_only``——自算值仅供展示与审计，永不参与保护。

策略集合随代码版本冻结，带一个指纹（:data:`POLICY_FINGERPRINT`）。实时
保护、回放与审计共用同一份策略；回放时若记录的指纹与当前不一致，必须在
核对结果里标红，而不是静默按新策略重算历史。
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

SELF_PESSIMUM = "self_pessimum"
CAL_ONLY = "cal_only"
SOURCES = (SELF_PESSIMUM, CAL_ONLY)

# 部位固定枚举。吸热器本体按集总模型管理时整体归入 PANEL。
PANEL = "panel"
HEADER = "header"
SUPPORT = "support"
PARTS = (PANEL, HEADER, SUPPORT)


@dataclass(frozen=True)
class PartPolicy:
    """单个部位的保护来源策略。"""

    part: str
    source: str
    rationale: str

    def as_dict(self) -> dict:
        return {"part": self.part, "source": self.source, "rationale": self.rationale}


#: 冻结的部位策略表。修改这张表属于策略改版，指纹随之变化。
PART_POLICIES: tuple[PartPolicy, ...] = (
    PartPolicy(PANEL, SELF_PESSIMUM, "受热管屏有独立功率/温升观测，自算仅在更保守时接管"),
    PartPolicy(HEADER, CAL_ONLY, "集箱无独立热工观测，自算不可辨识，只按标定保护"),
    PartPolicy(SUPPORT, CAL_ONLY, "支撑结构按机械温度标定保护，自算不参与"),
)


def _fingerprint() -> str:
    body = "|".join(f"{item.part}:{item.source}" for item in PART_POLICIES)
    return sha256(body.encode("utf-8")).hexdigest()[:12]


#: 策略指纹，写入每条保护判定记录，供回放核对。
POLICY_FINGERPRINT = _fingerprint()

_POLICY_BY_PART = {item.part: item for item in PART_POLICIES}


def policy_for(part: str) -> PartPolicy:
    """返回某部位的冻结策略；未知部位按最保守处理（cal_only）并拒绝猜测。"""

    item = _POLICY_BY_PART.get(part)
    if item is None:
        raise KeyError(f"未登记的吸热器部位: {part}")
    return item


def uses_self_params(part: str) -> bool:
    """该部位的保护是否允许自算值参与。"""

    return policy_for(part).source == SELF_PESSIMUM


def describe() -> list[dict]:
    """返回策略表（供控制台展示与落审计）。"""

    return [item.as_dict() for item in PART_POLICIES]
