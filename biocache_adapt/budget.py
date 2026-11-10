"""InfiniPot-V 的统一 KV 字节预算映射（纯计算、无模型导入）。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

BUDGET_TIERS = {"12.5%": 0.125, "25%": 0.25, "50%": 0.5}


@dataclass(frozen=True)
class KVSpec:
    num_layers: int = 28
    num_kv_heads: int = 4
    head_dim: int = 128
    dtype_bytes: int = 2
    tokens_per_frame: int = 140

    @property
    def bytes_per_token(self) -> int:
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.dtype_bytes


def parse_ratio(value) -> float:
    if isinstance(value, (int, float)):
        ratio = float(value)
    else:
        text = str(value).strip()
        ratio = float(text[:-1]) / 100.0 if text.endswith("%") else float(text)
    return ratio / 100.0 if ratio > 1 else ratio


def map_budget_to_infinipot(
    ratio,
    total_frames: int,
    *,
    block_size: int = 32,
    spec: KVSpec = KVSpec(),
    compression_method: str = "infinipot-v",
) -> dict:
    ratio = parse_ratio(ratio)
    if not 0 < ratio <= 1:
        raise ValueError("ratio 必须在 (0, 1]")
    if total_frames <= 0:
        raise ValueError("total_frames 必须为正")
    keep_frames = max(1, min(total_frames, round(total_frames * ratio)))
    block_size = max(int(block_size), keep_frames + 1)
    raw_bytes = total_frames * spec.tokens_per_frame * spec.bytes_per_token
    estimated = keep_frames * spec.tokens_per_frame * spec.bytes_per_token
    return {
        "method": "infinipot-v",
        "budget_ratio": ratio,
        "raw_kv_bytes": raw_bytes,
        "budget_bytes": int(raw_bytes * ratio),
        "hyper": {
            "block_size": block_size,
            "compress_frame_num": keep_frames,
            "compression_method": compression_method,
        },
        "estimated_persistent_bytes": estimated,
        "estimated_ratio_achieved": estimated / raw_bytes,
        "spec": {
            "num_layers": spec.num_layers,
            "num_kv_heads": spec.num_kv_heads,
            "head_dim": spec.head_dim,
            "dtype_bytes": spec.dtype_bytes,
            "tokens_per_frame": spec.tokens_per_frame,
        },
    }


def standard_budget_mappings(total_frames: int, **kwargs) -> dict:
    return {
        tier: map_budget_to_infinipot(ratio, total_frames, **kwargs)
        for tier, ratio in BUDGET_TIERS.items()
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="InfiniPot-V KV 预算映射（不加载模型）")
    parser.add_argument("--ratio", required=True, choices=tuple(BUDGET_TIERS))
    parser.add_argument("--total-frames", required=True, type=int)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--tokens-per-frame", type=int, default=140)
    args = parser.parse_args(argv)
    result = map_budget_to_infinipot(
        args.ratio,
        args.total_frames,
        block_size=args.block_size,
        spec=KVSpec(tokens_per_frame=args.tokens_per_frame),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
