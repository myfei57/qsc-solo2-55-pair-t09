"""平台装配与控制循环。

``Platform`` 把镜场、吸热器、熔盐、蒸汽、清洗、告警与审计按真实调用链装到
一起，并暴露四件事：操作编排（追日、循环建立、停堆、清洗、切换）、控制循环、
并发探测与状态快照。控制循环每个周期推进熔盐与吸热器、推进蒸汽与调度执行值、
判定告警并落审计、采集趋势。
"""

from __future__ import annotations

import dataclasses
import math
import threading

from .alarm.escalation import EscalationPolicy
from .alarm.history import AlarmHistory
from .alarm.latch import FaultLatch
from .alarm.rules import AlarmEvent, AlarmRules, default_rules
from .audit.log import AuditLog
from .audit.record import AuditRecord
from .clock import Clock, SystemClock
from .config import PlatformConfig, default_config
from .errors import PlantError
from .eventbus import Event, EventBus
from .field.field import FieldController
from .field.persist import FieldPersistence
from .heliostat.aim import AimPoint
from .heliostat.registry import Heliostat, HeliostatRegistry
from .heliostat.sun import SunModel, SunPosition
from .heliostat.zone import ZoneGuard
from .ids import IdFactory
from .ns.registry import NamespaceRegistry
from .ns.topology import Topology, build_default_topology
from .ns.zones import SafeZone, ZoneRegistry
from .receiver.derate import DerateController
from .receiver.flow import FlowGate
from .receiver.header import SaltHeader
from .receiver.persist import ReceiverPersistence
from .receiver.policy import PANEL, POLICY_FINGERPRINT, describe as describe_part_policies
from .receiver.protection import ENGINE_VERSION, ProtectionModel
from .receiver.receiver import Receiver
from .receiver.replay import (
    DERATE_CLEARED,
    DERATE_ENTERED,
    PARAMS_COMMITTED,
    PARAMS_RECALIBRATED,
    verify_protection_records,
)
from .receiver.self_params import CalibratedParams, SelfParamError
from .receiver.sensor import TemperatureSensor
from .receiver.thermal import LOSS_KW_K, RECEIVER_CAPACITY_KJ_K, ThermalModel
from .receiver.threshold import ThresholdModel
from .receiver.valve import InletValve
from .salt.loop import SaltLoop
from .salt.persist import SaltPersistence
from .sequence import (
    acquire_and_track,
    circulate,
    emergency_trip,
    reset_after_trip,
    shutdown_plant,
    startup_plant,
    wash_mirrors,
)
from .steam.bypass import BypassValve
from .steam.pressure import SteamPressure
from .steam.system import SteamSystem
from .steam.turbine import Turbine
from .steam.turbine_load import LoadRamp
from .store.jsonfile import JsonStore
from .store.journal import JsonlJournal
from .store.metrics import StoreMetrics
from .telemetry import TrendBuffer
from .temp.ambient import AmbientModel
from .temp.profile import TemperatureProfile
from .temp.solar import ClearSkyProfile
from .temp.temp import build_report
from .wash.cycle import WashCycle
from .wash.drain import DrainSystem
from .wash.truck import WashTruck

DEFAULT_UNIT = "field1"
DEFAULT_MIRROR = "m021"
AIM_POINT_Z_M = 180.0
MAINTENANCE_REACH_M = 24.0
MANUAL_HOLD_S = 900.0

# 镜场布置：三个分区按半径排开，方位角自西向东铺开。
_RING_SPECS: tuple[tuple[str, float, float, float, int], ...] = (
    ("a", 180.0, -60.0, 60.0, 41),
    ("b", 260.0, -75.0, 75.0, 41),
    ("c", 340.0, -85.0, 85.0, 41),
)

# 避让区布置：两处检修区与一处扫描区，落在 a 区镜子的检修落点附近。
_ZONE_SPECS: tuple[tuple[str, float, float, float, str], ...] = (
    ("maint-north", 0.0, 152.0, 26.0, "maintenance"),
    ("maint-west", -152.0, 62.0, 26.0, "maintenance"),
    ("scan-east", 150.0, 40.0, 30.0, "scan"),
)


class Platform:
    """塔式光热电站镜场与吸热器控制平台。"""

    def __init__(
        self,
        config: PlatformConfig | None = None,
        clock: Clock | None = None,
        unit: str = DEFAULT_UNIT,
    ) -> None:
        self.config = config or default_config()
        self.clock = clock or SystemClock()
        self.bus = EventBus()
        self.unit = unit

        store = JsonStore(self.config.storage_root)
        self.store = store
        self.metrics = StoreMetrics(store.root)
        self.audit = AuditLog(
            self.clock, self.bus, JsonlJournal(store.root, "audit", segment_limit=512)
        )
        self.namespaces = NamespaceRegistry(unit)
        self.topology: Topology = build_default_topology()
        self.trend = TrendBuffer(period_s=self.config.irradiance.sample_period_s)
        self.ids = IdFactory("op")

        self.zones = ZoneRegistry()
        self.heliostats = HeliostatRegistry()
        self._build_field()
        self.field = FieldController(
            self.heliostats,
            self.config.field,
            self.config.heliostat,
            self.config.irradiance,
            self.zones,
            ZoneGuard(self.zones, self.config.field.safe_zone_margin_m),
            FieldPersistence(store),
            AimPoint(0.0, 0.0, AIM_POINT_Z_M),
            maintenance_reach_m=MAINTENANCE_REACH_M,
            manual_hold_s=MANUAL_HOLD_S,
        )

        threshold = ThresholdModel(self.config.salt)
        thermal = ThermalModel(self.config.receiver)
        calibrated = CalibratedParams(RECEIVER_CAPACITY_KJ_K, LOSS_KW_K, 1.0)
        receiver_persistence = ReceiverPersistence(store)
        try:
            param_state, migrated, fresh = receiver_persistence.load_or_init(calibrated)
        except SelfParamError as error:
            # 自算参数版本无法解释：fail-closed 退回纯标定，并立即落审计告警。
            param_state, migrated, fresh = None, [], True
            self_recover_error = error
        else:
            self_recover_error = None
        protection = ProtectionModel(
            self.config.receiver,
            param_state.calibrated if param_state is not None else calibrated,
            part=PANEL,
            derate_lead_s=self.config.receiver.derate_lead_s,
            inlet_c=thermal.inlet_c,
            state=param_state,
        )
        self.receiver = Receiver(
            self.config.receiver,
            InletValve(self.config.receiver),
            FlowGate(self.config.receiver),
            TemperatureSensor(self.config.receiver.ambient_c),
            thermal,
            DerateController(self.config.receiver),
            SaltHeader(self.config.salt),
            threshold,
            protection=protection,
        )
        self.receiver_persistence = receiver_persistence
        self.self_param_migrations: list[int] = list(migrated)
        if self_recover_error is not None:
            self.bus.publish(
                Event.of(
                    "receiver",
                    "receiver",
                    "self_params_load_failed",
                    severity="alarm",
                    message="自算参数版本无法解释，保护已退回纯标定（fail-closed）",
                    detail=self_recover_error.message,
                )
            )
        elif migrated:
            self.bus.publish(
                Event.of(
                    "receiver",
                    "receiver",
                    "self_params_migrated",
                    severity="notice",
                    message="自算参数已按迁移链升级到当前版本",
                    applied=migrated,
                    revision=protection.state.revision,
                )
            )
        self.salt = SaltLoop(
            self.config.salt,
            threshold,
            thermal,
            self.receiver.header,
            SaltPersistence(store),
        )

        bypass = BypassValve(self.config.steam)
        ramp = LoadRamp(self.config.steam)
        self.steam = SteamSystem(
            self.config.steam,
            SteamPressure(self.config.steam),
            ramp,
            bypass,
            Turbine(self.config.steam, ramp, bypass),
        )

        self.sun_model = SunModel()
        self.solar = ClearSkyProfile(peak_dni_w_m2=950.0)
        self.ambient = AmbientModel()
        self.temperature_profile = TemperatureProfile(
            self.config.receiver,
            self.solar,
            self.ambient,
            self.heliostats.total_area_m2(),
            self.config.heliostat.reflectivity,
            inlet_c=self.receiver.thermal.inlet_c,
        )

        self.latch = FaultLatch(self.config.alarm)
        self.rules = AlarmRules(default_rules(self.config))
        self.history = AlarmHistory()
        self.escalation = EscalationPolicy(self.config.alarm)
        self.truck = WashTruck(self.config.wash)
        self.drain = DrainSystem(self.config.wash)
        self.wash_cycle = WashCycle()

        self._cycles = 0
        self._operations = 0
        self._lock = threading.RLock()
        self._register_namespaces()
        self._restore()

    # ---- 装配 ---------------------------------------------------------

    def _build_field(self) -> None:
        """按环形布置登记定日镜，并登记避让区。"""

        area = self.config.heliostat.mirror_area_m2
        index = 0
        for group, radius, start, end, count in _RING_SPECS:
            step = (end - start) / (count - 1)
            for position in range(count):
                index += 1
                theta = math.radians(start + step * position)
                self.heliostats.register(
                    Heliostat(
                        name=f"m{index:03d}",
                        x_m=round(radius * math.sin(theta), 3),
                        y_m=round(radius * math.cos(theta), 3),
                        z_m=2.5,
                        area_m2=area,
                        group=group,
                    )
                )
        for name, x_m, y_m, radius_m, kind in _ZONE_SPECS:
            self.zones.add(SafeZone(name, x_m, y_m, radius_m, kind))

    def _register_namespaces(self) -> None:
        """登记现场设备命名空间（控制台与审计共用同一套名字）。"""

        for area, device, kind in (
            ("field", "irradiance", "irradiance-meter"),
            ("field", "tracking", "heliostat-field"),
            ("field", "dispatch", "dispatch-controller"),
            ("tower", "receiver", "receiver"),
            ("tower", "inlet_valve", "control-valve"),
            ("salt", "cold_tank", "salt-tank"),
            ("salt", "hot_tank", "salt-tank"),
            ("salt", "cold_pump", "salt-pump"),
            ("salt", "hot_pump", "salt-pump"),
            ("steam", "bypass", "bypass-valve"),
            ("steam", "turbine", "steam-turbine"),
            ("wash", "truck", "wash-truck"),
            ("wash", "drain", "drain-valve"),
        ):
            self.namespaces.register(area, device, kind)

    def _restore(self) -> None:
        """从落盘状态恢复平台计数，并回放最近一次停堆原因。"""

        state = self.store.read("platform_state")
        if isinstance(state, dict):
            self._operations = int(state.get("operations", 0))
            self._cycles = int(state.get("cycles", 0))
        try:
            records = self.audit.to_records(self.audit.replay_from_journal(limit=100_000))
        except PlantError:
            return
        trips = [item for item in records if item.action == "receiver_trip_complete"]
        if trips:
            last = trips[-1]
            reason = str(last.payload.get("reason", "unknown"))
            self.latch.trip("receiver", reason, last.ts)

    def _persist(self) -> None:
        """落盘平台计数，供重启后对照。"""

        self.store.write(
            "platform_state",
            {"operations": self._operations, "cycles": self._cycles, "unit": self.unit},
        )

    def _record_operation(self, op_id: str, operation: str, report: dict) -> dict:
        """把编排步骤逐条写审计，并记一条汇总事件。"""

        asset = str(report.get("mirror", report.get("asset", "")))
        for step in report.get("steps", []):
            reserved = {"asset", "op_id", "severity", "message", "ts", "component", "action", "type"}
            detail = {
                key: value
                for key, value in dict(step.get("detail", {})).items()
                if not isinstance(value, (dict, list))
                and key not in reserved
            }
            self.bus.publish(
                Event.of(
                    step["component"],
                    step["component"],
                    step["action"],
                    asset=asset,
                    op_id=op_id,
                    severity="notice",
                    **detail,
                )
            )
        for violation in report.get("violations", []):
            self.bus.publish(
                Event.of(
                    "audit",
                    "audit",
                    "interlock_rejected",
                    asset=asset,
                    op_id=op_id,
                    severity="alarm",
                    message=f"{operation} 被拒绝：{violation}",
                    violation=violation,
                )
            )
        summary = {
            "op_id": op_id,
            "operation": operation,
            "asset": asset,
            "ok": bool(report.get("ok")),
            "violations": list(report.get("violations", [])),
        }
        if report.get("reason"):
            summary["reason"] = report["reason"]
        self.bus.publish(
            Event.of(
                "audit",
                "audit",
                f"{operation}_complete",
                severity="notice" if report.get("ok") else "warning",
                message=f"{operation} 完成" if report.get("ok") else f"{operation} 未完成",
                **summary,
            )
        )
        return summary

    # ---- 自算保护 -----------------------------------------------------

    def _publish_protection_events(self, heat: dict) -> None:
        """把自算收紧与降额进入/退出按版本化载荷落审计（回放就靠这些字段）。"""

        protection = self.receiver.protection
        if protection is None:
            return
        if heat.get("revision_committed"):
            observation = heat.get("self_params", {})
            committed = observation.get("committed", {})
            raw = observation.get("raw", {})
            self.receiver_persistence.save(protection.state)
            self.bus.publish(
                Event.of(
                    "receiver",
                    "receiver",
                    PARAMS_COMMITTED,
                    severity="warning",
                    message="自算热工参数收紧，降额保护按更保守口径生效",
                    revision=protection.state.revision,
                    schema=protection.state.schema,
                    engine_version=ENGINE_VERSION,
                    policy_fingerprint=POLICY_FINGERPRINT,
                    part=heat.get("part"),
                    committed=committed,
                    raw=raw,
                )
            )
        derate = heat.get("derate", {})
        changed = derate.get("changed")
        if changed in ("entered", "cleared"):
            action = DERATE_ENTERED if changed == "entered" else DERATE_CLEARED
            self.bus.publish(
                Event.of(
                    "receiver",
                    "receiver",
                    action,
                    severity="warning" if changed == "entered" else "notice",
                    message=(
                        "进入降额：" + self._explain_derate(derate)
                        if changed == "entered"
                        else "退出降额，回差确认温度已回落"
                    ),
                    decision=derate,
                    effective=protection.effective_params().as_dict(),
                    committed=protection.state.committed.as_dict(),
                    calibrated=protection.state.calibrated.as_dict(),
                    derate_lead_s=protection.status()["derate_lead_s"],
                )
            )

    @staticmethod
    def _explain_derate(derate: dict) -> str:
        """把结构化归因翻成一句可在审计正文里读的话。"""

        explanation = derate.get("explanation", {})
        if not explanation:
            return "触发降额"
        earlier = explanation.get("earlier_vs_calibration_s", 0.0)
        rate = explanation.get("projected_rate_c_s", 0.0)
        text = f"按外推温升 {rate:.4f} ℃/s，比纯标定提前 {earlier:.1f} s"
        vs_previous = explanation.get("vs_previous")
        if vs_previous:
            direction = {"earlier": "更早", "later": "更晚", "same": "持平"}.get(
                vs_previous.get("direction"), "持平"
            )
            text += f"；比上次{direction} {abs(vs_previous.get('lead_delta_s', 0.0)):.1f} s"
        factors = [item for item in explanation.get("factors", []) if abs(item["ratio"] - 1.0) > 0.01]
        if factors:
            names = {"capacity_kj_per_k": "热容", "loss_kw_per_k": "散热", "flow_gain": "流量计增益"}
            text += "；依据：" + "、".join(
                f"{names.get(item['factor'], item['factor'])}={item['self']}" for item in factors
            )
        text += f"（参数版本 r{explanation.get('revision', 0)}）"
        return text

    def recalibrate_receiver(
        self,
        capacity_kj_per_k: float,
        loss_kw_per_k: float,
        flow_gain: float = 1.0,
        operator: str = "maintenance",
    ) -> dict:
        """检修后重新标定吸热器热工参数（自算棘轮唯一的放松入口）。"""

        with self._lock:
            protection = self.receiver.protection
            if protection is None:
                raise SelfParamError("未挂版本化保护模型", component="receiver")
            state = protection.recalibrate(
                capacity_kj_per_k, loss_kw_per_k, flow_gain, self.clock.now(), operator
            )
            self.receiver_persistence.save(state)
            self.bus.publish(
                Event.of(
                    "receiver",
                    "receiver",
                    PARAMS_RECALIBRATED,
                    severity="notice",
                    message="吸热器热工参数检修后重新标定，自算棘轮已按新基线放松",
                    revision=state.revision,
                    schema=state.schema,
                    engine_version=ENGINE_VERSION,
                    policy_fingerprint=POLICY_FINGERPRINT,
                    operator=operator,
                    calibrated=state.calibrated.as_dict(),
                )
            )
            return protection.status()

    def degrade_receiver(
        self,
        *,
        capacity_kj_per_k: float | None = None,
        loss_kw_per_k: float | None = None,
        flow_gain: float | None = None,
    ) -> dict:
        """注入一次真值侧漂移（仿真/验证用：结焦、保温老化、流量计漂移）。"""

        with self._lock:
            self.receiver.thermal.degrade(
                capacity_kj_per_k=capacity_kj_per_k,
                loss_kw_per_k=loss_kw_per_k,
                flow_gain=flow_gain,
            )
            return {
                "capacity_kj_per_k": self.receiver.thermal.capacity_kj_per_k,
                "loss_kw_per_k": self.receiver.thermal.loss_kw_per_k,
                "flow_gain": self.receiver.thermal.flow_gain,
            }

    def verify_protection_replay(self, limit: int = 100_000) -> dict:
        """从落盘审计流水按当时版本回放保护判定并核对一致性。"""

        with self._lock:
            raw = self.audit.replay_from_journal(limit=limit)
            verdict = verify_protection_records(raw, self.config.receiver)
            return verdict.as_dict()

    # ---- 生命周期 -----------------------------------------------------

    def start(self) -> dict:
        """启动平台：核对拓扑与避让区登记。"""

        with self._lock:
            report = self.topology.verify(set())
            return {
                "unit": self.unit,
                "topology_missing": report["missing"],
                "mirrors": self.heliostats.describe(),
                "zones": self.zones.describe(),
                "noon_sun": self.sun_model.noon().as_dict(),
                "operations": self._operations,
            }

    def stop(self) -> dict:
        """停运平台：汽机卸载并走旁路，然后收起镜场、停泵关母管。"""

        with self._lock:
            op_id = self.ids.next()
            report = shutdown_plant(
                self.field, self.salt, self.receiver, self.steam, self.clock.now()
            )
            summary = self._record_operation(op_id, "shutdown", report)
            self._operations += 1
            self._persist()
            return {**report, **summary}

    def start_plant(
        self, dni_w_m2: float, load_mw: float, minute_of_day: float | None = None
    ) -> dict:
        """按控制链顺序把平台从冷态带到带负荷。"""

        with self._lock:
            op_id = self.ids.next()
            sun = self.sun_at(minute_of_day)
            report = startup_plant(
                self.field,
                self.salt,
                self.receiver,
                self.steam,
                sun,
                dni_w_m2,
                self.clock.now(),
                load_mw,
            )
            summary = self._record_operation(op_id, "startup", report)
            self._operations += 1
            self._persist()
            return {**report, **summary}

    def sun_at(self, minute_of_day: float | None = None) -> SunPosition:
        """返回给定时刻的太阳位置；默认按当前挂钟时间推算。"""

        if minute_of_day is None:
            minute_of_day = (self.clock.now() / 60.0) % 1440.0
        return self.sun_model.at(minute_of_day)

    # ---- 操作编排 -----------------------------------------------------

    def track(self, dni_w_m2: float, minute_of_day: float | None = None) -> dict:
        """追日：采集 → 落盘 → 判定 → 下发。"""

        with self._lock:
            op_id = self.ids.next()
            sun = self.sun_at(minute_of_day)
            report = acquire_and_track(
                self.field, dni_w_m2, sun, self.clock.now(), latched=self.latch.latched()
            )
            summary = self._record_operation(op_id, "track", report)
            self._operations += 1
            self._persist()
            return {**report, **summary}

    def circulate(self, target_flow_kg_s: float | None = None) -> dict:
        """建立熔盐循环：启泵 → 流量确认 → 通盐。"""

        with self._lock:
            op_id = self.ids.next()
            report = circulate(self.salt, self.receiver, 1.0, target_flow_kg_s)
            summary = self._record_operation(op_id, "circulate", report)
            self._operations += 1
            self._persist()
            return {**report, **summary}

    def changeover(self) -> dict:
        """冷热罐切换：先停冷泵、再启热泵。"""

        with self._lock:
            op_id = self.ids.next()
            handoff = self.salt.changeover()
            report = {
                "ok": handoff.ok,
                "steps": handoff.steps,
                "violations": handoff.violations,
                "peak_pressure_mpa": handoff.peak_pressure_mpa,
                "max_pumps_running": handoff.max_pumps_running,
            }
            summary = self._record_operation(op_id, "changeover", report)
            self._operations += 1
            self._persist()
            return {**report, **summary}

    def wash(self, mirrors: list[str] | None = None, zone: str = "maint-north") -> dict:
        """按批次清洗镜面。"""

        with self._lock:
            op_id = self.ids.next()
            batch = list(mirrors) if mirrors else self.field.sweeps.plan(self.heliostats.names())[0]
            report = wash_mirrors(
                self.field, self.wash_cycle, self.truck, self.drain, batch, zone, self.clock.now()
            )
            summary = self._record_operation(op_id, "wash", report)
            self._operations += 1
            self._persist()
            return {**report, **summary}

    def trip(self, reason: str = "overtemperature") -> dict:
        """停堆保护：闩锁 → 镜场收起 → 汽机旁路 → 关母管。"""

        with self._lock:
            op_id = self.ids.next()
            report = emergency_trip(
                self.receiver,
                self.field,
                self.steam,
                self.latch,
                self.history,
                self.clock.now(),
                reason,
            )
            summary = self._record_operation(op_id, "receiver_trip", report)
            self._operations += 1
            self._persist()
            return {**report, **summary}

    def reset_trip(self, operator: str = "shift-lead") -> dict:
        """复位停堆：先清闩锁，再复位吸热器。"""

        with self._lock:
            op_id = self.ids.next()
            report = reset_after_trip(self.receiver, self.latch, self.clock.now(), operator)
            summary = self._record_operation(op_id, "reset_trip", report)
            self._operations += 1
            self._persist()
            return {**report, **summary}

    def manual_aim(
        self,
        mirror: str = DEFAULT_MIRROR,
        azimuth_deg: float = 180.0,
        elevation_deg: float = 35.0,
        operator: str = "console",
        hold_s: float | None = None,
    ) -> dict:
        """手动接管一台定日镜。"""

        with self._lock:
            lease = self.field.manual_aim(
                mirror, azimuth_deg, elevation_deg, operator, self.clock.now(), hold_s
            )
            self.bus.publish(
                Event.of(
                    "field",
                    "field",
                    "manual_aim",
                    asset=mirror,
                    severity="notice",
                    message=f"{operator} 手动接管 {mirror}",
                    azimuth_deg=round(lease.azimuth_deg, 3),
                    elevation_deg=round(lease.elevation_deg, 3),
                )
            )
            return lease.as_dict(self.clock.now())

    def release_manual(self, mirror: str = DEFAULT_MIRROR) -> dict:
        """释放手动接管，把镜面交回自动追日。"""

        with self._lock:
            released = self.field.release_manual(mirror, self.clock.now())
            self.bus.publish(
                Event.of(
                    "field",
                    "field",
                    "manual_release",
                    asset=mirror,
                    severity="notice",
                    message=f"{mirror} 交回自动追日",
                    released=released,
                )
            )
            return {"mirror": mirror, "released": released}

    def expire_manual(self) -> dict:
        """清理过期的手动租约。"""

        with self._lock:
            gone = self.field.expire_manual(self.clock.now())
            return {"expired": gone, "remaining": self.field.manual.count()}

    def calibrate_flow(self, measured_kg_s: float, operator: str = "console") -> dict:
        """按实测流量重新标定基线，联锁门限同步改变。"""

        with self._lock:
            record = self.salt.calibrate_flow(measured_kg_s, self.clock.now(), operator)
            self.bus.publish(
                Event.of(
                    "salt",
                    "salt",
                    "flow_recalibrated",
                    severity="notice",
                    message="流量基线重新标定",
                    baseline_kg_s=round(record.baseline_kg_s, 3),
                    threshold_kg_s=round(record.threshold_kg_s, 3),
                    epoch=record.epoch,
                )
            )
            return record.as_dict()

    def dispatch(self, pct: float, writer: str = "dispatch") -> dict:
        """登记调度负荷并按本地保护上限求值。"""

        with self._lock:
            ceiling = self.receiver.ceiling_pct(self.salt.flow_kg_s())
            outcome = self.field.dispatch_load(pct, writer, ceiling)
            self.receiver.open_valve(outcome.effective_pct)
            return {**outcome.as_dict(), "flow_kg_s": round(self.salt.flow_kg_s(), 3)}

    def disturb_steam(self, delta_mpa: float) -> dict:
        """注入一次主蒸汽压力扰动。"""

        with self._lock:
            return self.steam.disturb(delta_mpa)

    def disturb_salt(self, heat_fraction: float) -> dict:
        """注入一次熔盐热负荷扰动。"""

        with self._lock:
            return self.salt.disturb(heat_fraction)

    # ---- 控制循环 -----------------------------------------------------

    def cycle(self, dt: float = 1.0) -> dict:
        """推进一个控制周期，返回本周期摘要。"""

        if dt <= 0:
            raise ValueError("步长必须为正")
        with self._lock:
            self._cycles += 1
            moment = self.clock.now()
            salt_report = self.salt.step(dt)
            power_kw = self.field.thermal_power_kw()
            dispatch_pct = self.field.step_dispatch(dt)
            if self.receiver.header.state() == "admitted":
                heat = self.receiver.heat(power_kw, self.salt.flow_kg_s(), dt, moment)
                self._publish_protection_events(heat)
            else:
                heat = {
                    "ok": True,
                    "state": self.receiver.state(),
                    "temperature_c": round(self.receiver.output_temperature_c(), 2),
                    "power_kw": round(power_kw, 2),
                    "derate": {"active": False},
                    "flow_verdict": {"ok": True},
                    "violations": [],
                    "skipped": "not_admitted",
                }
            load_target_mw = self.config.steam.load_max_mw * dispatch_pct / 100.0
            self.steam.track_load(load_target_mw)
            steam_report = self.steam.update(power_kw, dt)
            alarms = self._evaluate_alarms(moment)
            self.trend.append("receiver_temperature", moment, self.receiver.output_temperature_c())
            self.trend.append("salt_flow_kg_s", moment, self.salt.flow_kg_s())
            self.trend.append("steam_pressure_mpa", moment, self.steam.pressure.value_mpa())
            self.trend.append("field_power_kw", moment, power_kw)
            self._persist()
            return {
                "cycle": self._cycles,
                "salt": salt_report,
                "heat": heat,
                "steam": steam_report,
                "dispatch_pct": round(dispatch_pct, 3),
                "alarms": [event.as_dict() for event in alarms],
                "state": self.receiver.state(),
            }

    def _signals(self) -> dict:
        """汇总告警判定需要的现场信号。"""

        return {
            "receiver_temperature_c": self.receiver.output_temperature_c(),
            "header_pressure_mpa": self.receiver.header.pressure_mpa(),
            "cold_tank_pct": self.salt.cold_tank.level_pct(),
            "hot_tank_pct": self.salt.hot_tank.level_pct(),
            "steam_pressure_mpa": self.steam.pressure.value_mpa(),
            "salt_flow_kg_s": self.salt.flow_kg_s(),
        }

    def _evaluate_alarms(self, now: float) -> list[AlarmEvent]:
        """判定告警、写历史与审计，并在越限时触发停堆。"""

        events = self.rules.evaluate(self._signals())
        if not events:
            return []
        self.history.record_all(now, events)
        for event in events:
            occurrences = self.history.occurrences(event.rule, now - self.escalation.window_s)
            escalated = self.escalation.evaluate(occurrences, event.severity)
            self.bus.publish(
                Event.of(
                    "alarm",
                    "alarm",
                    event.rule,
                    severity=event.severity,
                    message=event.message,
                    value=round(event.value, 3),
                    threshold=round(event.threshold, 3),
                    level=escalated.level,
                    occurrences=occurrences,
                )
            )
        overtemperature = any(item.rule == "receiver-overtemperature" for item in events)
        if overtemperature and not self.latch.latched():
            self.trip("overtemperature")
        return events

    def concurrency_probe(self, rounds: int = 24) -> dict:
        """用真实并行调用探测调度与本地保护的仲裁结果。

        两个线程分别代表自动调度与本地联锁，同时提交负荷指令；探测只读取最终
        执行值，用来确认共享执行器始终收敛到更保守的那个上限。
        """

        if rounds < 1:
            raise ValueError("探测轮数必须为正")
        ceiling = self.config.field.dispatch_rate_pct_s * 10.0

        def worker(writer: str, pct: float, limit: float) -> None:
            for _ in range(rounds):
                self.field.dispatch_load(pct, writer, limit)

        threads = [
            threading.Thread(target=worker, args=("auto_dispatch", 100.0, 100.0)),
            threading.Thread(target=worker, args=("local_interlock", 100.0, ceiling)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        outcome = self.field.dispatch.resolve(ceiling)
        self.receiver.open_valve(outcome.effective_pct)
        return {
            "rounds": rounds,
            "requested_pct": round(outcome.requested_pct, 3),
            "ceiling_pct": round(outcome.ceiling_pct, 3),
            "effective_pct": round(outcome.effective_pct, 3),
            "clamped": outcome.clamped,
            "writer": outcome.writer,
            "committed_on_valve": round(self.receiver.valve.target_pct(), 3),
            "writers": ["auto_dispatch", "local_interlock"],
        }

    # ---- 查询 ---------------------------------------------------------

    def temperature_report(self) -> dict:
        """返回温度与降额报告。"""

        return build_report(
            self.config.receiver,
            self.receiver.output_temperature_c(),
            self.receiver.derate.active(),
        ).as_dict()

    def forecast(self, minute_of_day: float | None = None, horizon_min: float = 60.0) -> dict:
        """返回未来一段时间的出口温度预估。"""

        moment = (self.clock.now() / 60.0) % 1440.0 if minute_of_day is None else float(minute_of_day)
        flow = self.salt.flow_kg_s() or self.config.salt.flow_baseline_kg_s
        points = self.temperature_profile.predict(moment, flow, horizon_min=horizon_min)
        return {
            "minute_of_day": round(moment, 1),
            "flow_kg_s": round(flow, 3),
            "ambient_mean_c": round(self.ambient.mean_c(), 2),
            "energy_remaining_kj_m2": round(self.solar.energy_remaining_kj_m2(moment), 2),
            "points": [point.as_dict() for point in points],
            "minutes_to_derate": self.temperature_profile.minutes_to_derate(moment, flow),
        }

    def alert_levels(self) -> list[dict]:
        """返回每条告警规则在当前窗口内的升级等级。"""

        now = self.clock.now()
        report: list[dict] = []
        for rule in self.rules.rules:
            occurrences = self.history.occurrences(rule.name, now - self.escalation.window_s)
            escalated = self.escalation.evaluate(occurrences, rule.severity)
            report.append({"rule": rule.name, **escalated.as_dict()})
        return report

    def status(self) -> dict:
        """汇总状态快照，供控制台与命令行读取。"""

        return {
            "platform": self.config.name,
            "unit": self.unit,
            "operations": self._operations,
            "cycles": self._cycles,
            "now": round(self.clock.now(), 3),
            "utc": self.clock.utc_iso(),
            "field": self.field.status(),
            "receiver": self.receiver.snapshot().as_dict(),
            "receiver_heat_steps": self.receiver.heat_steps(),
            "derate_state": {
                "entries": (
                    self.receiver.protection.entries()
                    if self.receiver.has_versioned_protection()
                    else self.receiver.derate.entries()
                ),
                "clears": (
                    self.receiver.protection.clears()
                    if self.receiver.has_versioned_protection()
                    else self.receiver.derate.clears()
                ),
                "evaluations": (
                    self.receiver.protection.evaluation_count()
                    if self.receiver.has_versioned_protection()
                    else self.receiver.derate.evaluation_count()
                ),
                "min_ceiling_pct": round(
                    (
                        self.receiver.protection.min_ceiling_pct()
                        if self.receiver.has_versioned_protection()
                        else self.receiver.derate.min_ceiling_pct()
                    ),
                    3,
                ),
                "last_temperature_c": round(
                    (
                        self.receiver.protection.last_temperature_c()
                        if self.receiver.has_versioned_protection()
                        else self.receiver.derate.last_temperature_c()
                    ),
                    2,
                ),
                "versioned": self.receiver.has_versioned_protection(),
            },
            "self_params": (
                self.receiver.protection.status()
                if self.receiver.has_versioned_protection()
                else {"versioned": False}
            ),
            "protection_policy": {"fingerprint": POLICY_FINGERPRINT, "engine_version": ENGINE_VERSION,
                                   "parts": describe_part_policies()},
            "flow_gate": {
                "hold_s": round(self.receiver.gate.hold_s(), 3),
                "resets": self.receiver.gate.resets(),
                "confirmations": self.receiver.gate.confirmations(),
                "last_flow_kg_s": round(self.receiver.gate.last_flow(), 3),
            },
            "thermal": {
                "inlet_c": self.receiver.thermal.inlet_c,
                "capacity_kj_per_k": self.receiver.thermal.capacity_kj_per_k,
                "loss_kw_per_k": self.receiver.thermal.loss_kw_per_k,
                "flow_gain": self.receiver.thermal.flow_gain,
                "updates": self.receiver.thermal.updates(),
                "dry_steps": self.receiver.thermal.dry_steps(),
                "peak_c": round(self.receiver.thermal.peak_c(), 2),
            },
            "valve_moves": self.receiver.valve.moves(),
            "temperature": self.temperature_report(),
            "salt": self.salt.status(),
            "steam": self.steam.status(),
            "steam_pressure_updates": self.steam.pressure.updates(),
            "bypass_held_open_s": round(self.steam.bypass.held_open_s(self.clock.now()), 3),
            "wash": self.truck.snapshot(),
            "drain": {
                "tray_m3": round(self.drain.tray_m3(), 4),
                "recovered_m3": round(self.drain.recovered_m3(), 4),
                "pump_running": self.drain.pump_running(),
                "premature_stops": self.drain.premature_stops(),
            },
            "alarm": {
                "latched": self.latch.latched(),
                "source": self.latch.source(),
                "reason": self.latch.reason(),
                "trips": self.latch.trips(),
                "resets": self.latch.resets(),
                "reset_rejections": self.latch.rejections(),
                "history": self.history.count(),
                "by_rule": self.history.by_rule(),
                "rules": self.rules.names(),
            },
            "levels": self.alert_levels(),
            "store": self.metrics.summary(),
            "audit": self.audit.summary(),
            "ids_issued": self.ids.issued(),
        }

    def snapshot(self) -> dict:
        """返回只读的整体快照（状态 + 拓扑 + 命名空间 + 避让区 + 趋势）。"""

        return {
            "status": self.status(),
            "topology": self.topology.summary(),
            "namespaces": self.namespaces.describe(),
            "zones": self.zones.describe(),
            "heliostats": self.heliostats.describe(),
            "rules": self.rules.describe(),
            "trend": self.trend.snapshot(),
            "alarms_recent": self.history.recent(10),
        }

    def mirrors_in_window(self) -> list[str]:
        """返回当前清洗窗口里的镜子。"""

        return self.field.sweeps.active_mirrors()

    def note_error(self, error: PlantError) -> dict:
        """把平台异常写入审计，返回落盘记录。"""

        payload = error.as_record()
        record = self.audit.record(
            AuditRecord.create(
                self.clock.now(),
                "",
                payload["component"],
                "error",
                severity="warning",
                message=payload["message"],
            )
        )
        return record

    def reset_state(self) -> bool:
        """清掉落盘的平台计数（交付前重新标定用）。"""

        if not self.store.exists("platform_state"):
            return False
        return self.store.remove("platform_state")


def build_plant(
    config: PlatformConfig | None = None,
    clock: Clock | None = None,
    storage_root: str | None = None,
    unit: str = DEFAULT_UNIT,
) -> Platform:
    """构造一个平台；``storage_root`` 可覆盖配置里的存储目录。"""

    base = config or default_config()
    if storage_root is not None:
        base = dataclasses.replace(base, storage_root=storage_root)
    return Platform(base, clock=clock, unit=unit)
