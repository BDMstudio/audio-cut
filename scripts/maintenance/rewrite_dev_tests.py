from pathlib import Path
from textwrap import dedent
path = Path('development.md')
lines = path.read_text(encoding='utf-8').splitlines()
start = None
end = None
for i,line in enumerate(lines):
    if line.startswith('## 6. 测试矩阵'):
        start = i
    elif start is not None and line.startswith('## 7.'):
        end = i
        break
if start is None or end is None:
    raise SystemExit('未找到测试矩阵块')
new_block = dedent('''
## 6. 测试矩阵
- unit：	est_cut_alignment, 	est_cutting_consistency, 	est_segment_labeling, 	est_pre_vocal_split, 	est_gpu_pipeline, 	est_chunk_feature_builder_gpu, 	est_chunk_feature_builder_stft_equivalence, 	est_silero_chunk_vad, 	est_pure_vocal_focus_windows, 	est_track_feature_cache, 	est_mdx23_path_resolution 等覆盖算法细节与配置解析。
- integration：	est_pipeline_v2_valley.py、	ests/test_pure_vocal_detection_v2.py 验证全流程与主要模式。
- contracts：	est_config_contracts.py、	ests/contracts/test_valley_contract.py 保证配置与 valley 行为。
- benchmarks：	ests/benchmarks/test_chunk_vs_full_equivalence.py 输出 chunk vs full 误差报告。
- performance：	ests/performance/test_valley_perf.py 监控 MDD+VPP 耗时。
- sanity：	ests/sanity/ort_mdx23_cuda_sanity.py 自检 GPU Provider。
- 慢测试与 GPU case 使用 @pytest.mark.slow、@pytest.mark.gpu 标记，CI 默认跳过。
''').splitlines()
lines = lines[:start] + new_block + lines[end:]
path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
