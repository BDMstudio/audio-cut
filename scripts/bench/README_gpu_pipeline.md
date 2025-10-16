<!-- File: scripts/bench/README_gpu_pipeline.md -->
<!-- AI-SUMMARY: 说明 GPU vs CPU 基准脚本的运行方式、输出字段和报告格式。 -->

# GPU 流水线基准报告说明

本文件描述 `run_gpu_cpu_baseline.py` 的使用方式、输出字段及如何在 PR 中附带报告。

## 运行脚本

```bash
python scripts/bench/run_gpu_cpu_baseline.py input/sample.wav --write-markdown
```

* 默认输出目录：`output/bench/<timestamp>/`。
* 脚本会分别运行 GPU/CPU 路径，写出 `gpu_cpu_baseline.json` 与 `gpu_cpu_baseline.md`。
* 可传入多个音频或目录，脚本将为每个样本生成独立子目录。

## JSON 字段

| 字段 | 说明 | 备注 |
| --- | --- | --- |
| `gpu.processing_time_s` / `cpu.processing_time_s` | 端到端耗时（秒） | 用于计算 speedup |
| `*_throughput_ratio` | `audio_duration_s / processing_time_s` | 值越大越好 |
| `summary.speedup_ratio` | (1 - gpu_time / cpu_time) | ≥0.30 视为达标 |
| `gpu.gpu_meta.h2d_ms` / `dtoh_ms` / `compute_ms` | H2D/DtoH/推理耗时（毫秒） | 需 ≥15% 改善 |
| `gpu.gpu_meta.peak_mem_bytes` | GPU 显存峰值（字节） | ≤ CPU 基线 +10% |
| `summary.meets_target` | 是否通过门槛 | `True`/`False` |

## Markdown 示例

`gpu_cpu_baseline.md` 为 PR 直接引用的表格：

```markdown
| file | cpu_time_s | gpu_time_s | throughput_cpu | throughput_gpu | speedup | meets_target |
| --- | --- | --- | --- | --- | --- | --- |
| sample.wav | 32.41 | 20.11 | 0.31 | 0.50 | 37.94% | ✅ |
```

## 附件建议

1. 将 `gpu_cpu_baseline.json` / `gpu_cpu_baseline.md` 一并上传至 `output/bench/<timestamp>/` 并纳入 PR。
2. `tests/benchmarks/test_chunk_vs_full_equivalence.py::test_chunk_vs_full_equivalence_real_model` 生成的 `chunk_vs_full_real.{json,md}` 可作为契约测试佐证。
3. 关键指标（speedup、H2D/DtoH、显存峰值）需在 PR 描述中引用。
## �� GPU ̽���� strict ģʽ

`ash
python scripts/bench/run_multi_gpu_probe.py input/sample.wav --devices 0,1 --mode v2.2_mdd --output-root output/bench
# CPU ��֤�ɴ� --devices cpu�������ϸ� GPU ʧ�ܼ�������׷�� --strict-gpu��
`

* ���д�� output/bench/multi_gpu_probe_<timestamp>/multi_gpu_probe.json��devices[*].gpu_meta �� gpu_pipeline_* ָ�꣨processed_chunks��peak_mem_bytes��H2D/DtoH �ȣ���
* --strict-gpu ��� gpu_pipeline.strict_gpu ��Ϊ 	rue��EnhancedVocalSeparator ������ GPU �쳣ʱ������� CPU������ѹ�����Ի������ա�
* ������ un_gpu_cpu_baseline.py ����һ������ PR�����ǵ���/�࿨���ϸ�ģʽ����·����
