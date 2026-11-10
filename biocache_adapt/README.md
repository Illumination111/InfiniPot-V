# InfiniPot-V BioCache streaming adapter

该包在不复制模型实现的前提下，复用官方 `qwen_inference_ovu.py` 的视觉 block
编码/生成阶段与 `kvcache_utils.py::process_kv_cache`，提供统一流式视频 QA 接口。
模型依赖仅在 `load()` 或真实帧处理时导入。

## 接口与状态

```python
adapter.open_stream(video_id)
adapter.push_frames(timestamped_frames, visible_until=query.query_time)
response = adapter.answer(query)  # `query()` 为别名
adapter.close_stream()
```

- `push_frames` 只处理外部传入的当前可见帧，不接受视频路径、不保存未来帧。
- 视频 `DynamicCache` 跨 query 持续存在并按官方 InfiniPot-V 方法压缩。
- 问题后缀和生成 token 写入视频 cache 的深拷贝，回答后丢弃，避免污染后续 query。
- `persistent_state/static_extra_state/temporary_state` 分别导出视频 cache、新增静态态、
  query 临时 cache；`tensor_nbytes` 递归计量真实张量。
- 提供阶段计时、12.5%/25%/50% 预算映射、manifest、统一 JSONL schema，以及
  OVO/StreamingBench/TwinPast/VNBench/EgoLifeQA 字段 loader。

## 支持边界与风险

- 官方仓库原实现面向离线完整视频；本适配层把同一 block 逻辑用于外部帧批。
  跨 batch 的 Qwen M-RoPE 时间偏移、processor 对内存帧列表的接受形式尚未做模型
  推理验证。调用方也可直接传含 `input_ids/pixel_values_videos/video_grid_thw` 的
  当前批预处理 mapping，避免适配层猜测解码格式。
- 低到只保留极少帧时，官方 `infinipot-v` 的 TaR 查询帧取整可能为零；这属于官方
  压缩函数边界，真实运行前应在目标配置上验证。
- 未进行模型推理、benchmark、权重或数据下载；静态检查不代表精度复现。

## 静态测试

```bash
python -m py_compile biocache_adapt/*.py
python biocache_adapt/budget.py --ratio 25% --total-frames 120
python -m biocache_adapt.cli --help
python -m biocache_adapt.cli --dry-run --budget-ratio 25% \
  --total-frames 120 --run-dir /tmp/infinipot-v-dry
```
