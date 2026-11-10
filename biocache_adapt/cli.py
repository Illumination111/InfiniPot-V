"""InfiniPot-V BioCache 适配层 dry-run CLI。"""

from __future__ import annotations

import argparse
import json
import os

from .budget import map_budget_to_infinipot
from .manifest import collect_manifest, write_manifest
from .protocol import NullAdapter, StreamQuery, TimelineDriver, TimestampedFrame

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="InfiniPot-V 统一流式协议 dry-run")
    parser.add_argument("--dry-run", action="store_true", help="不导入模型依赖、不运行推理")
    parser.add_argument("--budget-ratio", default="25%", choices=("12.5%", "25%", "50%"))
    parser.add_argument("--total-frames", type=int, default=120)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--tokens-per-frame", type=int, default=140)
    parser.add_argument("--run-dir", default="/tmp/infinipot-v-biocache-dry")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.dry_run:
        raise SystemExit(
            "CLI 仅用于 --dry-run；真实评测由外部解码器调用 InfiniPotVAdapter.push_frames"
        )
    from .budget import KVSpec

    budget = map_budget_to_infinipot(
        args.budget_ratio,
        args.total_frames,
        block_size=args.block_size,
        spec=KVSpec(tokens_per_frame=args.tokens_per_frame),
    )
    config = {"dry_run": True, "model_loaded": False, "budget": budget}
    write_manifest(args.run_dir, collect_manifest(REPO_ROOT, config))
    adapter = NullAdapter()
    query = StreamQuery(1.0, "dry-run question", ["x", "y"], "A", "dry-q")
    results = TimelineDriver().run(
        adapter,
        "dry-stream",
        [
            ("frames", [TimestampedFrame(0.0, None), TimestampedFrame(1.0, None)]),
            ("query", query),
        ],
    )
    output = {
        "dry_run": True,
        "model_loaded": False,
        "events": adapter.calls,
        "responses": [response for _, response in results],
        "budget": budget,
        "manifest": os.path.join(args.run_dir, "manifest.json"),
    }
    print(json.dumps(output, ensure_ascii=False))
    return output


if __name__ == "__main__":
    main()
