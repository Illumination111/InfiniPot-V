"""围绕官方 qwen_inference_ovu.py/kvcache_utils.py 的流式轻适配。"""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional

from .meter import StageTimer
from .protocol import StreamQuery, TimestampedFrame

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class InfiniPotHyper:
    block_size: int = 32
    compress_frame_num: int = 24
    compression_method: str = "infinipot-v"
    tar_ratio: float = 0.5
    query_ratio: float = 0.25
    adaptive_pooling: bool = False
    per_frame: bool = False
    max_new_tokens: int = 16

    def to_dict(self) -> dict:
        return asdict(self)


class InfiniPotVAdapter:
    """跨 query 保留视频 DynamicCache、隔离问题 token 的适配器。

    所有模型依赖都在 ``load``/真实调用内延迟导入。`push_frames` 只处理调用方
    此刻提供的帧，不接收视频路径，也不保存待处理的未来帧。
    """

    def __init__(self, hyper: Optional[InfiniPotHyper] = None, repo_root: str = REPO_ROOT):
        self.hyper = hyper or InfiniPotHyper()
        self.repo_root = repo_root
        self.model = None
        self.processor = None
        self._official = None
        self._video_cache = None
        self._query_cache = None
        self._stream_id = None
        self._opened = False
        self._last_timestamp = None
        self._frames_seen = 0
        self._system_size = None
        self._token_per_frame = None
        self._temporal_offset = 0
        self._last_position_ids = None
        self.timer = StageTimer()

    def load(self, model_path: str, *, verbose: bool = False) -> None:
        """使用官方 evaluator 构造模型；调用本函数才会导入 torch/transformers。"""
        if self.repo_root not in sys.path:
            sys.path.insert(0, self.repo_root)
        from qwen_inference_ovu import OfflineVideoEval

        self._official = OfflineVideoEval(
            model_path=model_path,
            max_frames_num=self.hyper.block_size,
            block_size=self.hyper.block_size,
            compress_frame_num=self.hyper.compress_frame_num,
            compression_method=self.hyper.compression_method,
            tar_ratio=self.hyper.tar_ratio,
            query_ratio=self.hyper.query_ratio,
            adaptive_pooling=self.hyper.adaptive_pooling,
            per_frame=self.hyper.per_frame,
            verbose=verbose,
        )
        self.model = self._official.model
        self.processor = self._official.processor

    def open_stream(self, stream_id: str, **metadata: Any) -> None:
        if self.model is None:
            raise RuntimeError("请先调用 load()；dry-run 不应调用 open_stream")
        self._stream_id = str(stream_id)
        self.reset()
        self._opened = True

    def reset(self) -> None:
        self._last_timestamp = None
        self._frames_seen = 0
        self._system_size = None
        self._token_per_frame = None
        self._temporal_offset = 0
        self._last_position_ids = None
        self._query_cache = None
        if self.model is not None:
            from transformers import DynamicCache

            self._video_cache = DynamicCache()

    @staticmethod
    def _unpack(item) -> tuple[float, Any]:
        if isinstance(item, TimestampedFrame):
            return float(item.timestamp), item.frame
        if isinstance(item, Mapping) and "timestamp" in item:
            return float(item["timestamp"]), item["frame"]
        timestamp, frame = item
        return float(timestamp), frame

    def _prepare_batch(self, frames: list) -> Mapping[str, Any]:
        """把当前可见帧批转成官方 block encoder 所需输入。"""
        if len(frames) == 1 and isinstance(frames[0], Mapping):
            required = {"input_ids", "pixel_values_videos", "video_grid_thw"}
            if required.issubset(frames[0]):
                return frames[0]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=[text], images=None, videos=[frames], padding=True, return_tensors="pt"
        )
        return inputs.to(self.model.device)

    def _encode_visible_batch(self, inputs: Mapping[str, Any]) -> None:
        import torch

        input_ids = inputs["input_ids"]
        video_token_id = self.model.config.video_token_id
        mask = (input_ids[0] == video_token_id).nonzero(as_tuple=True)[0]
        if mask.numel() == 0:
            raise ValueError("processor 输出中没有 video token")
        total_frames = int(inputs["video_grid_thw"][0, 0].item())
        token_per_frame = int(mask.numel() // total_frames)
        if self._token_per_frame is not None and token_per_frame != self._token_per_frame:
            raise ValueError("不同 push batch 的 token_per_frame 不一致")
        self._token_per_frame = token_per_frame

        video_embeds = self._official._visual_encoding_block(
            inputs["pixel_values_videos"],
            inputs["video_grid_thw"],
            0,
            total_frames,
            total_frames,
        )
        video_embeds = video_embeds.to(self.model.device)
        position_ids_full, _ = self.model.model.get_rope_index(
            input_ids, video_grid_thw=inputs["video_grid_thw"]
        )
        video_positions = position_ids_full[:, :, mask].clone()
        video_positions[0] += self._temporal_offset

        if self._system_size is None:
            self._system_size = int(mask[0].item())
            system_ids = input_ids[:, : self._system_size]
            system_embeds = self.model.model.get_input_embeddings()(system_ids)
            model_inputs = video_embeds.unsqueeze(0)
            model_inputs = torch.cat(
                [system_embeds, model_inputs.to(system_embeds.dtype)], dim=1
            )
            position_ids = torch.cat(
                [position_ids_full[:, :, : self._system_size], video_positions], dim=2
            )
        else:
            model_inputs = video_embeds.unsqueeze(0)
            position_ids = video_positions

        with self.timer.stage("video_prefill", frames=total_frames):
            outputs = self.model(
                position_ids=position_ids,
                past_key_values=self._video_cache,
                inputs_embeds=model_inputs,
                use_cache=True,
            )
        self._video_cache = outputs[1]
        self._last_position_ids = position_ids[:, :, -1:].clone()
        self._temporal_offset = int(video_positions[0].max().item()) + 1
        self._frames_seen += total_frames

        if self._frames_seen > self.hyper.compress_frame_num:
            if self.repo_root not in sys.path:
                sys.path.insert(0, self.repo_root)
            from kvcache_utils import process_kv_cache

            with self.timer.stage("cache_compress", frames_seen=self._frames_seen):
                self._video_cache, _ = process_kv_cache(
                    past_key_values=self._video_cache,
                    model=self.model,
                    system_size=self._system_size,
                    inst_size=0,
                    token_per_frame=self._token_per_frame,
                    compress_frame_num=self.hyper.compress_frame_num,
                    method=self.hyper.compression_method,
                    tar_ratio=self.hyper.tar_ratio,
                    query_ratio=self.hyper.query_ratio,
                    adaptive_pooling=self.hyper.adaptive_pooling,
                    is_first_block=self._frames_seen == total_frames,
                    is_last_block=False,
                    per_frame=self.hyper.per_frame,
                )

    def push_frames(
        self,
        frames: Iterable[TimestampedFrame],
        *,
        visible_until: Optional[float] = None,
    ) -> None:
        if not self._opened:
            raise RuntimeError("流尚未 open_stream")
        decoded = []
        last_timestamp = self._last_timestamp
        for item in frames:
            timestamp, frame = self._unpack(item)
            if visible_until is not None and timestamp > visible_until + 1e-9:
                raise ValueError(f"frame {timestamp} 超过 visible_until={visible_until}")
            if last_timestamp is not None and timestamp < last_timestamp - 1e-9:
                raise ValueError("frame timestamp 必须单调不减")
            last_timestamp = timestamp
            decoded.append(frame)
        if not decoded:
            return
        inputs = self._prepare_batch(decoded)
        self._encode_visible_batch(inputs)
        self._last_timestamp = last_timestamp

    def prefill_until(self, t: float, frames: Iterable[TimestampedFrame] = ()) -> None:
        self.push_frames(frames, visible_until=float(t))
        if self._last_timestamp is not None and self._last_timestamp > float(t) + 1e-9:
            raise ValueError("缓存已含目标时间之后的未来帧")

    @staticmethod
    def _prompt(query: StreamQuery) -> str:
        if not query.options:
            return query.question
        options = "\n".join(f"{chr(65 + i)}. {v}" for i, v in enumerate(query.options))
        return f"{query.question}\n{options}\nAnswer with the option letter only."

    def _question_inputs(self, query: StreamQuery) -> dict:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": ""},
                    {"type": "text", "text": self._prompt(query)},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.processor.tokenizer(text, return_tensors="pt").input_ids
        return {"input_ids": input_ids.to(self.model.device)}

    def answer(self, query: StreamQuery) -> str:
        if not self._opened:
            raise RuntimeError("流尚未 open_stream")
        if self._frames_seen == 0:
            raise RuntimeError("至少 push 一个可见帧后才能回答")
        if self._last_timestamp is not None and self._last_timestamp > query.query_time + 1e-9:
            raise ValueError("拒绝回答：缓存含 query_time 之后的未来帧")
        # 问题与生成 token 仅写入 clone；视频 cache 在 query 间保持不变。
        self._query_cache = copy.deepcopy(self._video_cache)
        old_limit = self._official.MAX_GEN_TOKENS
        self._official.MAX_GEN_TOKENS = self.hyper.max_new_tokens
        try:
            with self.timer.stage("query_answer", query_id=query.query_id):
                return str(
                    self._official._generation_stage(
                        self._question_inputs(query),
                        self._query_cache,
                        self._last_position_ids,
                    )
                )
        finally:
            self._official.MAX_GEN_TOKENS = old_limit
            self._query_cache = None

    query = answer

    def persistent_state(self) -> Mapping[str, Any]:
        return {"video_dynamic_cache": self._video_cache}

    @staticmethod
    def static_extra_state() -> Mapping[str, Any]:
        return {}

    def temporary_state(self) -> Mapping[str, Any]:
        return {"query_dynamic_cache": self._query_cache} if self._query_cache is not None else {}

    def close_stream(self) -> None:
        self._video_cache = None
        self._query_cache = None
        self._opened = False
        self._stream_id = None
        self._last_timestamp = None
        self._frames_seen = 0
