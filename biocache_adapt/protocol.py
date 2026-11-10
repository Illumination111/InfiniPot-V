"""统一因果流式视频 QA 协议（不负责视频解码）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional


@dataclass(frozen=True)
class TimestampedFrame:
    timestamp: float
    frame: Any


@dataclass
class StreamQuery:
    query_time: float
    question: str
    options: Optional[list] = None
    answer: Optional[str] = None
    query_id: str = ""
    meta: dict = field(default_factory=dict)


class StreamAdapterProtocol:
    def open_stream(self, stream_id: str, **metadata: Any) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def push_frames(
        self, frames: Iterable[TimestampedFrame], *, visible_until: Optional[float] = None
    ) -> None:
        raise NotImplementedError

    def prefill_until(self, t: float, frames: Iterable[TimestampedFrame] = ()) -> None:
        raise NotImplementedError

    def answer(self, query: StreamQuery) -> str:
        raise NotImplementedError

    def query(self, query: StreamQuery) -> str:
        return self.answer(query)

    def close_stream(self) -> None:
        raise NotImplementedError

    def persistent_state(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def static_extra_state(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def temporary_state(self) -> Mapping[str, Any]:
        raise NotImplementedError


class TimelineDriver:
    """消费已经按时间产生的事件；不排序或预读 iterable。"""

    def run(self, adapter, stream_id: str, events: Iterable[tuple], metadata=None) -> list:
        adapter.open_stream(stream_id, **dict(metadata or {}))
        results = []
        try:
            for kind, value in events:
                if kind == "frames":
                    adapter.push_frames(value)
                elif kind == "query":
                    results.append((value, adapter.answer(value)))
                else:
                    raise ValueError(f"未知事件 {kind!r}")
        finally:
            adapter.close_stream()
        return results


class NullAdapter(StreamAdapterProtocol):
    """无模型 dry-run 适配器。"""

    def __init__(self):
        self.calls = []
        self.last_timestamp = None

    def open_stream(self, stream_id: str, **metadata: Any) -> None:
        self.calls.append(("open_stream", stream_id))

    def reset(self) -> None:
        self.last_timestamp = None
        self.calls.append(("reset",))

    def push_frames(self, frames, *, visible_until=None) -> None:
        for item in frames:
            timestamp = float(item.timestamp if hasattr(item, "timestamp") else item[0])
            if visible_until is not None and timestamp > visible_until + 1e-9:
                raise ValueError("frame timestamp 超过 visible_until")
            if self.last_timestamp is not None and timestamp < self.last_timestamp - 1e-9:
                raise ValueError("frame timestamp 必须单调不减")
            self.last_timestamp = timestamp
            self.calls.append(("push_frame", timestamp))

    def prefill_until(self, t: float, frames=()) -> None:
        self.push_frames(frames, visible_until=t)
        self.calls.append(("prefill_until", float(t)))

    def answer(self, query: StreamQuery) -> str:
        if self.last_timestamp is not None and self.last_timestamp > query.query_time + 1e-9:
            raise ValueError("已接收 query_time 之后的未来帧")
        self.calls.append(("answer", query.query_id))
        return "A"

    query = answer

    def close_stream(self) -> None:
        self.calls.append(("close_stream",))

    def persistent_state(self) -> Mapping[str, Any]:
        return {}

    def static_extra_state(self) -> Mapping[str, Any]:
        return {}

    def temporary_state(self) -> Mapping[str, Any]:
        return {}
