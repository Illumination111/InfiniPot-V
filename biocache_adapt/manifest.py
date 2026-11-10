"""运行 manifest 与统一 JSONL schema。"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Optional

from .protocol import StreamQuery

INPUT_SCHEMA_VERSION = "biocache.streaming_qa.input.v1"
RESULT_SCHEMA_VERSION = "biocache.streaming_qa.result.v1"
SUPPORTED_DATASETS = ("ovobench", "streamingbench", "twinpast", "vnbench", "egolifeqa")


def _git(repo_root: str, *args) -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "-C", repo_root, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        return None


def _sha256(path: str) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def collect_manifest(repo_root: str, config: dict) -> dict:
    package_dir = os.path.join(repo_root, "biocache_adapt")
    sources = {}
    for name in sorted(os.listdir(package_dir)):
        if name.endswith(".py"):
            sources[name] = _sha256(os.path.join(package_dir, name))
    for name in ("qwen_inference_ovu.py", "kvcache_utils.py"):
        sources[name] = _sha256(os.path.join(repo_root, name))
    return {
        "schema": "biocache.manifest.v1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "argv": list(sys.argv),
        "python_version": sys.version,
        "config": config,
        "git": {
            "commit": _git(repo_root, "rev-parse", "HEAD"),
            "branch": _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git(repo_root, "status", "--porcelain")),
        },
        "source_hashes": sources,
    }


def write_manifest(run_dir: str, manifest: dict) -> str:
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path


def _first(record, *names, default=None):
    for name in names:
        if record.get(name) is not None:
            return record[name]
    return default


def _seconds(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).split(":")
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    return float(value)


def normalize_jsonl_record(record: dict, benchmark: str) -> dict:
    benchmark = benchmark.lower()
    if benchmark not in SUPPORTED_DATASETS:
        raise ValueError(f"不支持 benchmark={benchmark!r}")
    timestamp = _first(record, "query_time", "realtime", "timestamp", "time_stamp", "time")
    if timestamp is None:
        raise ValueError("缺少 query timestamp")
    options = _first(record, "options", "choices")
    if isinstance(options, dict):
        options = list(options.values())
    query = StreamQuery(
        query_time=_seconds(timestamp),
        question=str(_first(record, "question", "query", "prompt", default="")),
        options=list(options) if options is not None else None,
        answer=None
        if _first(record, "answer", "gt", "ground_truth", "label") is None
        else str(_first(record, "answer", "gt", "ground_truth", "label")),
        query_id=str(_first(record, "query_id", "question_id", "id", "qid", default="")),
        meta={"benchmark": benchmark, "source": record},
    )
    return {
        "schema": INPUT_SCHEMA_VERSION,
        "benchmark": benchmark,
        "video_id": str(_first(record, "video_id", "video", "video_name", default="")),
        "video_path": _first(record, "video_path", "path"),
        "duration": _first(record, "duration", "video_duration"),
        "query": query,
    }


def load_jsonl(path: str, benchmark: str) -> list:
    output = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                output.append(normalize_jsonl_record(json.loads(line), benchmark))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return output


def result_record(benchmark: str, video_id: str, query: StreamQuery, response: str, metrics=None):
    return {
        "schema": RESULT_SCHEMA_VERSION,
        "benchmark": benchmark,
        "video_id": video_id,
        "query": asdict(query),
        "response": response,
        "metrics": metrics or {},
    }


def append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
