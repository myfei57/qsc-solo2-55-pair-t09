"""吸热器自算参数持久化。

自算的生效参数与版本必须跨重启保留：重启后若回到出厂标定，已经因结焦
收紧的保护会悄悄放松，等于把棘轮拆掉。落盘内容包括标定值、保守生效值、
schema 与 revision；读盘时先跑版本迁移（见 :mod:`self_params`），
迁移记录交回调用方落审计。
"""

from __future__ import annotations

from ..store.jsonfile import JsonStore
from .self_params import (
    CalibratedParams,
    SelfParamState,
    initial_state,
    migrate_document,
    state_from_dict,
)

SELF_PARAMS_DOC = "receiver/self_params"


class ReceiverPersistence:
    """自算参数读写与启动迁移。"""

    def __init__(self, store: JsonStore) -> None:
        self._store = store

    @property
    def store(self) -> JsonStore:
        return self._store

    def save(self, state: SelfParamState) -> str:
        """落盘自算参数（原子写）。"""

        return self._store.write(SELF_PARAMS_DOC, state.as_dict())

    def load_or_init(self, calibrated: CalibratedParams, now: float = 0.0) -> tuple[SelfParamState, list[int], bool]:
        """读回自算状态；不存在则按标定初始化。

        返回 ``(状态, 迁移经过的 schema 列表, 是否为全新初始化)``。
        读到未知版本时抛 :class:`SelfParamError`，由装配方决定是否
        fail-closed 退回纯标定并告警。
        """

        raw = self._store.read(SELF_PARAMS_DOC)
        if not isinstance(raw, dict):
            return initial_state(calibrated, now), [], True
        migrated, applied = migrate_document(raw)
        state = state_from_dict(migrated)
        if applied:
            self.save(state)
        return state, applied, False

    def documents(self) -> list[str]:
        """返回吸热器侧已落盘的文档列表。"""

        return [doc for doc in self._store.list_ids() if doc == SELF_PARAMS_DOC]
