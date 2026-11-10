"""真实张量递归字节计量与分阶段计时。"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from typing import Any, Mapping, Optional


def _is_tensor(value: Any) -> bool:
    return (
        hasattr(value, "element_size")
        and callable(value.element_size)
        and hasattr(value, "nelement")
        and callable(value.nelement)
    )


def tensor_nbytes(value: Any, _seen: Optional[set] = None) -> int:
    """递归统计 Tensor/DynamicCache 中去重后的真实 storage 元素字节。"""
    if value is None:
        return 0
    if _seen is None:
        _seen = set()
    oid = id(value)
    if oid in _seen:
        return 0
    if _is_tensor(value):
        _seen.add(oid)
        return int(value.element_size()) * int(value.nelement())
    if isinstance(value, (str, bytes, int, float, bool)):
        return 0
    _seen.add(oid)
    total = 0
    if hasattr(value, "key_cache") and hasattr(value, "value_cache"):
        return tensor_nbytes(value.key_cache, _seen) + tensor_nbytes(value.value_cache, _seen)
    layers = getattr(value, "layers", None)
    if layers is not None and not isinstance(value, (list, tuple, dict)):
        for layer in layers:
            total += tensor_nbytes(getattr(layer, "keys", None), _seen)
            total += tensor_nbytes(getattr(layer, "values", None), _seen)
        return total
    if isinstance(value, Mapping):
        return sum(tensor_nbytes(item, _seen) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(tensor_nbytes(item, _seen) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return sum(tensor_nbytes(getattr(value, f.name), _seen) for f in fields(value))
    if hasattr(value, "__dict__"):
        return sum(tensor_nbytes(item, _seen) for item in vars(value).values())
    return total


class StageTimer:
    def __init__(self):
        self.records = []

    @contextmanager
    def stage(self, name: str, **metadata):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.records.append(
                {
                    "stage": name,
                    "wall_ms": (time.perf_counter() - started) * 1000.0,
                    "metadata": metadata,
                }
            )

    def summary(self) -> dict:
        totals = {}
        for record in self.records:
            item = totals.setdefault(record["stage"], {"calls": 0, "wall_ms": 0.0})
            item["calls"] += 1
            item["wall_ms"] += record["wall_ms"]
        return {"stages": totals, "records": list(self.records)}


def state_bytes(adapter) -> dict:
    persistent = tensor_nbytes(adapter.persistent_state())
    static = tensor_nbytes(adapter.static_extra_state())
    temporary = tensor_nbytes(adapter.temporary_state())
    return {
        "persistent_bytes": persistent,
        "static_extra_bytes": static,
        "temporary_bytes": temporary,
        "total_cost_bytes": persistent + static,
    }
