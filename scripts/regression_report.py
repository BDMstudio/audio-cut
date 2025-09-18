#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scripts/regression_report.py
# AI-SUMMARY: 批量运行无缝分割（smart/v2 统一指挥中心），基于 config.yaml 的
#             segment_min_duration/segment_max_duration 进行越界统计，
#             生成 JSON + Markdown 回归报告（不引入任何新依赖）。

import os
import sys
import json
import time
import math
import statistics
from pathlib import Path
from datetime import datetime

# 将项目 src 加入路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from vocal_smart_splitter.core.seamless_splitter import SeamlessSplitter
from vocal_smart_splitter.utils.config_manager import get_config

AUDIO_EXTS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg'}


def find_audio_files(input_dir: Path):
    for p in input_dir.rglob('*'):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def collect_metrics_from_result(result: dict, min_dur: float, max_dur: float) -> dict:
    segments_meta = result.get('segments', [])
    durations = [float(seg.get('duration', 0.0)) for seg in segments_meta]
    zero_len = sum(1 for d in durations if d <= 0.0)
    below_min = sum(1 for d in durations if d < min_dur)
    above_max = sum(1 for d in durations if d > max_dur)

    metrics = {
        'success': bool(result.get('success', False)),
        'processing_type': result.get('processing_type') or result.get('method', ''),
        'num_segments': int(result.get('num_segments', len(segments_meta))),
        'perfect_reconstruction': bool(result.get('seamless_validation', {}).get('perfect_reconstruction', False)),
        'length_match': bool(result.get('seamless_validation', {}).get('length_match', False)),
        'max_difference': float(result.get('seamless_validation', {}).get('max_difference', float('nan')))
            if 'seamless_validation' in result else float('nan'),
        'rms_difference': float(result.get('seamless_validation', {}).get('rms_difference', float('nan')))
            if 'seamless_validation' in result else float('nan'),
        'segment_durations_sec': durations,
        'duration_median_sec': float(statistics.median(durations)) if durations else 0.0,
        'duration_mean_sec': float(statistics.mean(durations)) if durations else 0.0,
        'duration_min_sec': float(min(durations)) if durations else 0.0,
        'duration_max_sec': float(max(durations)) if durations else 0.0,
        'zero_length_segments': int(zero_len),
        'below_min_count': int(below_min),
        'above_max_count': int(above_max),
    }
    return metrics


def main():
    input_dir = ROOT / 'input'
    if not input_dir.exists():
        print(f"[ERR] 输入目录不存在: {input_dir}")
        sys.exit(1)

    # 读取边界仅使用 config.yaml 的 quality_control.segment_min/max
    min_dur = float(get_config('quality_control.segment_min_duration', 5.0))
    max_dur = float(get_config('quality_control.segment_max_duration', 18.0))

    # 初始化分割器
    sample_rate = int(get_config('audio.sample_rate', 44100))
    splitter = SeamlessSplitter(sample_rate=sample_rate)

    # 输出目录：一次运行一个汇总根目录，每首歌一个子目录
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_root = ensure_dir(ROOT / 'output' / f'regression_{ts}')

    all_items = list(find_audio_files(input_dir))
    results = []

    for idx, audio_path in enumerate(all_items, 1):
        rel = audio_path.relative_to(input_dir)
        song_dir = ensure_dir(out_root / rel.parent / rel.stem)
        print(f"[{idx}/{len(all_items)}] 处理: {rel}")
        t0 = time.time()
        try:
            res = splitter.split_audio_seamlessly(str(audio_path), str(song_dir), mode='smart_split')
        except Exception as e:
            res = {'success': False, 'error': str(e), 'input_file': str(audio_path), 'output_dir': str(song_dir)}
        elapsed = time.time() - t0

        metrics = collect_metrics_from_result(res, min_dur, max_dur)
        record = {
            'input_file': str(audio_path),
            'relative': str(rel),
            'output_dir': str(song_dir),
            'elapsed_sec': float(elapsed),
            **metrics,
        }
        results.append(record)

    # 汇总统计
    total = len(results)
    ok = sum(1 for r in results if r['success'])
    perfect = sum(1 for r in results if r['perfect_reconstruction'])
    zero_len_files = sum(1 for r in results if r['zero_length_segments'] > 0)
    any_below = sum(1 for r in results if r['below_min_count'] > 0)
    any_above = sum(1 for r in results if r['above_max_count'] > 0)

    summary = {
        'total_files': total,
        'success_files': ok,
        'perfect_reconstruction_files': perfect,
        'files_with_zero_length_segments': zero_len_files,
        'files_with_segments_below_min': any_below,
        'files_with_segments_above_max': any_above,
        'min_duration_sec': min_dur,
        'max_duration_sec': max_dur,
        'generated_at': ts,
    }

    report = {
        'summary': summary,
        'results': results,
    }

    # 输出 JSON + Markdown
    json_path = out_root / 'regression_report.json'
    md_path = out_root / 'regression_summary.md'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Markdown 摘要（仅客观统计；不引入任何新的阈值概念）
    lines = []
    lines.append(f"# 回归摘要 @ {ts}")
    lines.append("")
    lines.append(f"- 样本总数: {summary['total_files']}")
    lines.append(f"- 成功处理: {summary['success_files']}")
    lines.append(f"- 完美重构(拼接校验): {summary['perfect_reconstruction_files']}")
    lines.append(f"- 含0秒片段的文件数: {summary['files_with_zero_length_segments']}")
    lines.append(f"- 含<min片段的文件数: {summary['files_with_segments_below_min']}")
    lines.append(f"- 含>max片段的文件数: {summary['files_with_segments_above_max']}")
    lines.append(f"- 边界(min,max): ({min_dur}, {max_dur}) 秒")
    lines.append("")
    lines.append("## 明细（每文件）")
    for r in results:
        lines.append(f"- {r['relative']} | segs={r['num_segments']} | median={r['duration_median_sec']:.2f}s | mean={r['duration_mean_sec']:.2f}s | <min={r['below_min_count']} | >max={r['above_max_count']} | perfect={r['perfect_reconstruction']}")

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"完成。报告输出: {json_path}")
    print(f"摘要: {md_path}")


if __name__ == '__main__':
    main()

