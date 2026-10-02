"""吸热器。

吸热器把入射热功率转成熔盐温升，并守住三条边界：没有确认流量不得通盐（否则
干烧）、出口温度进入降额带必须压低负荷上限、越过上限必须跳闸并闩锁。阀门、
流量门控、温度模型、降额控制器与母管接口各自独立，顺序由编排保证。

版本化自算保护（:mod:`protection`）用实际功率、流量与温升在线辨识热容、散热
与流量计增益，按部位冻结的策略决定自算/标定来源：受热管屏采用自算悲观值，
集箱等部位只按标定；自算参数带 schema/revision，改版即迁移，实时、回放、
审计共用同一版本解释。
"""

from __future__ import annotations

from .derate import DerateController, DerateState
from .estimator import Estimate, ThermalEstimator
from .flow import FlowGate, FlowStatus
from .header import SaltHeader, HeaderState
from .persist import ReceiverPersistence
from . import policy
from .protection import Decision, Observation, ProtectionModel
from .receiver import AT_POWER, CIRCULATING, COLD, DERATED, TRIPPED, Receiver
from .replay import (
    DERATE_CLEARED,
    DERATE_ENTERED,
    PARAMS_COMMITTED,
    PARAMS_RECALIBRATED,
    ReplayVerdict,
    interpret_decision,
    verify_protection_records,
)
from .sensor import TemperatureSensor
from .self_params import (
    SCHEMA_VERSION,
    CalibratedParams,
    ParamSet,
    SelfParamError,
    SelfParamState,
    initial_state,
    is_more_conservative,
    migrate_document,
    pessimistic,
    state_from_dict,
)
from .snapshot import ReceiverSnapshot
from .thermal import SALT_CP_KJ_KG_K, INLET_TEMP_C, LOSS_KW_K, RECEIVER_CAPACITY_KJ_K, ThermalModel
from .threshold import ThresholdModel
from .valve import InletValve

__all__ = [
    "DerateController",
    "DerateState",
    "FlowGate",
    "FlowStatus",
    "SaltHeader",
    "HeaderState",
    "Receiver",
    "COLD",
    "CIRCULATING",
    "AT_POWER",
    "DERATED",
    "TRIPPED",
    "TemperatureSensor",
    "ReceiverSnapshot",
    "ThermalModel",
    "ThresholdModel",
    "InletValve",
    "SALT_CP_KJ_KG_K",
    "INLET_TEMP_C",
    "LOSS_KW_K",
    "RECEIVER_CAPACITY_KJ_K",
    "ProtectionModel",
    "Decision",
    "Observation",
    "ThermalEstimator",
    "Estimate",
    "CalibratedParams",
    "ParamSet",
    "SelfParamState",
    "SelfParamError",
    "SCHEMA_VERSION",
    "initial_state",
    "pessimistic",
    "is_more_conservative",
    "migrate_document",
    "state_from_dict",
    "ReceiverPersistence",
    "policy",
    "DERATE_ENTERED",
    "DERATE_CLEARED",
    "PARAMS_COMMITTED",
    "PARAMS_RECALIBRATED",
    "ReplayVerdict",
    "interpret_decision",
    "verify_protection_records",
]
