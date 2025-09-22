# -*- coding: utf-8 -*-
# File: scripts/validate_segments_integrity.py
# AI-SUMMARY: 校验分割输出目录：统计并对比（1）整首分离人声全文件，（2）带伴奏混音片段总和，（3）分离人声片段总和的时长与样本数；可选对比原音频。

from __future__ import annotations
import os
import sys
import glob
import math
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import soundfile as sf


# --- Path helpers -----------------------------------------------------------
def _repo_root() -> str:
    # scripts/ sits directly under project root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def _default_output_base() -> str:
    # Always resolve to repo_root/output regardless of current working dir
    return os.path.join(_repo_root(), 'output')


@dataclass
class AudioStat:
    path: str
    samples: int
    sr: int

    @property
    def seconds(self) -> float:
        return self.samples / float(self.sr) if self.sr > 0 else 0.0


def read_len(path: str) -> AudioStat:
    data, sr = sf.read(path, always_2d=False)
    if hasattr(data, 'shape'):
        n = int(data.shape[0])
    else:
        n = len(data)
    return AudioStat(path=path, samples=n, sr=int(sr))


def human_time(sec: float) -> str:
    if not math.isfinite(sec):
        return 'N/A'
    m, s = divmod(max(sec, 0.0), 60.0)
    h, m = divmod(int(m), 60)
    return f"{h:d}:{m:02d}:{s:05.2f}" if h else f"{m:02d}:{s:05.2f}"


def collect_stats(out_dir: str):
    # 1) 混音片段（带伴奏）
    mix_seg_paths = sorted(glob.glob(os.path.join(out_dir, 'segment_*.wav')))
    mix_stats = [read_len(p) for p in mix_seg_paths]

    # 2) 分离人声片段
    vocal_seg_dir = os.path.join(out_dir, 'segments_vocal')
    vocal_seg_paths = sorted(glob.glob(os.path.join(vocal_seg_dir, 'segment_*_vocal.wav')))
    vocal_stats = [read_len(p) for p in vocal_seg_paths]

    # 3) 整首分离人声
    vocal_full_paths = sorted(glob.glob(os.path.join(out_dir, '*_vocal_full.wav')))
    vocal_full = read_len(vocal_full_paths[0]) if vocal_full_paths else None

    # 4) 整首器乐（可选）
    inst_paths = sorted(glob.glob(os.path.join(out_dir, '*_instrumental.wav')))
    inst_full = read_len(inst_paths[0]) if inst_paths else None

    return mix_stats, vocal_stats, vocal_full, inst_full


def sum_stats(stats: List[AudioStat]) -> Tuple[int, Optional[int]]:
    if not stats:
        return 0, None
    srs = {st.sr for st in stats}
    # 如果采样率不一致，仍统计样本数，但标记不一致
    total_samples = sum(st.samples for st in stats)
    sr = stats[0].sr if len(srs) == 1 else None
    return total_samples, sr


def _list_candidate_output_dirs(base_dir: str) -> list[tuple[str, float]]:
    if not os.path.isdir(base_dir):
        return []
    candidates = []
    for name in os.listdir(base_dir):
        p = os.path.join(base_dir, name)
        if not os.path.isdir(p):
            continue
        # 判定规则：满足任一即可认为是有效输出目录
        has_mix = bool(glob.glob(os.path.join(p, 'segment_*.wav')))
        has_vocal_segs = bool(glob.glob(os.path.join(p, 'segments_vocal', 'segment_*_vocal.wav')))
        has_vocal_full = bool(glob.glob(os.path.join(p, '*_vocal_full.wav')))
        if has_mix or has_vocal_segs or has_vocal_full:
            try:
                mtime = os.path.getmtime(p)
            except Exception:
                mtime = 0.0
            candidates.append((p, mtime))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def _interactive_pick_output_dir() -> Optional[str]:
    base = _default_output_base()
    cands = _list_candidate_output_dirs(base)
    if not cands:
        print(f'[ERROR] 未发现可用输出目录：{base} 下没有包含 segment_*.wav 的子目录')
        return None

    print('请选择要校验的输出目录:')
    for idx, (p, m) in enumerate(cands, start=1):
        mix_cnt = len(glob.glob(os.path.join(p, 'segment_*.wav')))
        vocal_cnt = len(glob.glob(os.path.join(p, 'segments_vocal', 'segment_*_vocal.wav')))
        print(f'  {idx:2d}) {os.path.basename(p)}   mix={mix_cnt:3d}  vocal={vocal_cnt:3d}')
    print('  q) 退出')

    sel = input('输入序号回车（默认 1 ）: ').strip()
    if sel.lower() in ('q', 'quit', 'exit'):
        return None
    if sel == '':
        return cands[0][0]
    try:
        k = int(sel)
        if 1 <= k <= len(cands):
            return cands[k - 1][0]
    except Exception:
        pass
    print('[ERROR] 输入无效。')
    return None


def main():
    ap = argparse.ArgumentParser(description='验证分割输出目录的时长守恒与对比')
    ap.add_argument('--dir', required=False, help='分割输出目录（包含 segment_*.wav 等）。若不提供，将进入交互选择。')
    ap.add_argument('--orig', help='原始混音音频（可选，用于对比）')
    ap.add_argument('--tolerance-samples', type=int, default=1, help='判定守恒时允许的样本误差（默认1样本）')
    ap.add_argument('--analyze-tails', action='store_true', help='额外分析每个片段结尾的“近零静音”长度统计')
    ap.add_argument('--eps', type=float, default=1e-6, help='近零判定阈值 |x|<=eps（默认1e-6）')
    ap.add_argument('--max-tail-ms', type=int, default=200, help='最多检查尾部多少毫秒（默认200ms）')
    ap.add_argument('--reconstruct', action='store_true', help='将MIX片段串接并与原始音频逐样本对比（需要 --orig）')
    args = ap.parse_args()

    if args.dir:
        out_dir = os.path.abspath(args.dir)
        if not os.path.isdir(out_dir):
            # 允许仅给目录名，自动从 repo_root/output 下解析
            cand = os.path.join(_default_output_base(), args.dir)
            if os.path.isdir(cand):
                out_dir = cand
    else:
        out_dir = _interactive_pick_output_dir()
        if not out_dir:
            sys.exit(2)

    if not os.path.isdir(out_dir):
        print(f'[ERROR] 目录不存在: {out_dir}')
        sys.exit(2)

    mix_stats, vocal_stats, vocal_full, inst_full = collect_stats(out_dir)

    mix_total_samples, mix_sr = sum_stats(mix_stats)
    vocal_total_samples, vocal_sr = sum_stats(vocal_stats)

    print('=' * 80)
    print(f'[DIR] {out_dir}')
    print('-' * 80)
    print(f'MIX segments:        count={len(mix_stats):3d}  total={mix_total_samples:9d} samples'
          + ('' if mix_sr is None else f' @{mix_sr}Hz')
          + f'  ({human_time(mix_total_samples / (mix_sr or 1) if mix_sr else 0.0)})')
    print(f'VOCAL segments:      count={len(vocal_stats):3d}  total={vocal_total_samples:9d} samples'
          + ('' if vocal_sr is None else f' @{vocal_sr}Hz')
          + f'  ({human_time(vocal_total_samples / (vocal_sr or 1) if vocal_sr else 0.0)})')

    if vocal_full:
        print(f'VOCAL full:          path={os.path.basename(vocal_full.path):40s} '
              f'len={vocal_full.samples:9d} samples @{vocal_full.sr}Hz  ({human_time(vocal_full.seconds)})')
    else:
        print('VOCAL full:          [not found]')

    if inst_full:
        print(f'INSTRUMENTAL full:   path={os.path.basename(inst_full.path):40s} '
              f'len={inst_full.samples:9d} samples @{inst_full.sr}Hz  ({human_time(inst_full.seconds)})')

    if args.orig:
        if os.path.isfile(args.orig):
            orig = read_len(args.orig)
            print(f'ORIGINAL mix:        path={os.path.basename(orig.path):40s} '
                  f'len={orig.samples:9d} samples @{orig.sr}Hz  ({human_time(orig.seconds)})')
        else:
            print(f'ORIGINAL mix:        [not found] {args.orig}')
            orig = None
    else:
        orig = None

    print('-' * 80)
    # 对比：
    # 1) MIX 片段总和 vs VOCAL FULL（仅作参考，采样率可能不同；严格来说应与 ORIGINAL 对比）
    if vocal_full and mix_sr is not None:
        mix_sec = mix_total_samples / float(mix_sr)
        diff = mix_sec - vocal_full.seconds
        print(f'DIFF: sum(MIX segments) - VOCAL full = {diff:+.6f}s')

    # 2) VOCAL 片段总和 vs VOCAL FULL（同一声道体系期望接近）
    if vocal_full and vocal_sr is not None:
        vocal_sec = vocal_total_samples / float(vocal_sr)
        diff = vocal_sec - vocal_full.seconds
        ok = abs(diff) * (vocal_sr or 1) <= args.tolerance_samples
        print(f'DIFF: sum(VOCAL segments) - VOCAL full = {diff:+.6f}s   ' + ('PASS' if ok else 'FAIL'))

    # 3) sum(MIX segments) vs ORIGINAL（如提供）
    if orig and mix_sr is not None:
        mix_sec = mix_total_samples / float(mix_sr)
        diff = mix_sec - orig.seconds
        ok = abs(diff) * (mix_sr or 1) <= args.tolerance_samples
        print(f'DIFF: sum(MIX segments) - ORIGINAL    = {diff:+.6f}s   ' + ('PASS' if ok else 'FAIL'))

    # 4) sum(VOCAL segments) vs ORIGINAL（如提供）
    if orig and vocal_sr is not None:
        vocal_sec = vocal_total_samples / float(vocal_sr)
        diff = vocal_sec - orig.seconds
        print(f'DIFF: sum(VOCAL segments) - ORIGINAL  = {diff:+.6f}s')
    print('=' * 80)

    if args.analyze_tails:
        analyze_tails(out_dir, eps=args.eps, max_check_ms=args.max_tail_ms)

    if args.reconstruct and args.orig:
        reconstruct_and_compare(out_dir, args.orig)


# --- Advanced analysis -------------------------------------------------------
def _tail_silence_len_samples(x, eps=1e-6, max_check=None):
    n = len(x)
    if n == 0:
        return 0
    if max_check is not None:
        start = max(0, n - int(max_check))
    else:
        start = 0
    i = n - 1
    cnt = 0
    while i >= start:
        v = float(x[i])
        if -eps <= v <= eps:
            cnt += 1
            i -= 1
            continue
        break
    return cnt


def _db(x):
    import numpy as _np
    if x <= 0:
        return -120.0
    return 20.0 * float(_np.log10(x))


def analyze_tails(out_dir: str, eps: float = 1e-6, max_check_ms: int = 200):
    """Report per-segment tail near-silence length (<=eps) for MIX and VOCAL segments."""
    print('[TailAnalysis] eps=', eps, 'max_check_ms=', max_check_ms)
    mix_seg_paths = sorted(glob.glob(os.path.join(out_dir, 'segment_*.wav')))
    vocal_seg_paths = sorted(glob.glob(os.path.join(out_dir, 'segments_vocal', 'segment_*_vocal.wav')))
    total_mix_tail = total_vocal_tail = 0
    max_check = None
    sr_guess = None
    if mix_seg_paths:
        s = read_len(mix_seg_paths[0]); sr_guess = s.sr
        max_check = int(max_check_ms/1000.0 * s.sr)
    print('MIX tails:')
    for p in mix_seg_paths:
        d, sr = sf.read(p, always_2d=False)
        n = len(d)
        tail = _tail_silence_len_samples(d, eps=eps, max_check=max_check)
        total_mix_tail += tail
        rms = float((d[-min(1024, n):]**2).mean()**0.5) if n else 0.0
        msg = (
            f"  {os.path.basename(p):30s} tail_silent={tail:5d} samples  "
            f"({tail/(sr or 1):.4f}s)  tail_rms={_db(rms):.1f} dB"
        )
        print(msg)
    print(
        f"[MIX] total_tail_silent={total_mix_tail} samples  "
        f"({(total_mix_tail/(sr_guess or 1)):.6f}s)"
    )

    print('VOCAL tails:')
    for p in vocal_seg_paths:
        d, sr = sf.read(p, always_2d=False)
        n = len(d)
        tail = _tail_silence_len_samples(d, eps=eps, max_check=max_check)
        total_vocal_tail += tail
        rms = float((d[-min(1024, n):]**2).mean()**0.5) if n else 0.0
        msg = (
            f"  {os.path.basename(p):30s} tail_silent={tail:5d} samples  "
            f"({tail/(sr or 1):.4f}s)  tail_rms={_db(rms):.1f} dB"
        )
        print(msg)
    print(
        f"[VOCAL] total_tail_silent={total_vocal_tail} samples  "
        f"({(total_vocal_tail/(sr_guess or 1)):.6f}s)"
    )

# --- Reconstruction check ----------------------------------------------------
def reconstruct_and_compare(out_dir: str, orig_path: str, eps: float = 0.0):
    import numpy as np
    mix_seg_paths = sorted(glob.glob(os.path.join(out_dir, 'segment_*.wav')))
    if not mix_seg_paths:
        print('[Reconstruct] no mix segments found')
        return
    parts = []
    sr_ref = None
    for p in mix_seg_paths:
        d, sr = sf.read(p, always_2d=False)
        if sr_ref is None:
            sr_ref = sr
        elif sr != sr_ref:
            print(f'[Reconstruct][WARN] sample rate mismatch in {p}: {sr} vs {sr_ref}')
        parts.append(d.astype('float64', copy=False))
    concat = np.concatenate(parts) if parts else np.zeros(0, dtype='float64')
    orig, sr_o = sf.read(orig_path, always_2d=False)
    if sr_o != sr_ref:
        print(f'[Reconstruct][WARN] original sr {sr_o} != segments sr {sr_ref}, comparing by min length')
    n = min(len(concat), len(orig))
    if n == 0:
        print('[Reconstruct] empty signals')
        return
    diff = concat[:n].astype('float64') - orig[:n].astype('float64')
    max_abs = float(np.max(np.abs(diff)))
    print('[Reconstruct] len_concat=', len(concat), ' len_orig=', len(orig), ' common=', n)
    print(f'[Reconstruct] max |concat - orig| = {max_abs:.3e}')
    if len(concat) != len(orig):
        delta = (len(concat) - len(orig)) / float(sr_ref or 44100)
        print(f'[Reconstruct] length delta (sec) = {delta:+.6f}s')


if __name__ == '__main__':
    main()

