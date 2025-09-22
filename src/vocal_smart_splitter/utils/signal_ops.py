# -*- coding: utf-8 -*-
# File: src/vocal_smart_splitter/utils/signal_ops.py
# AI-SUMMARY: 提供基础信号处理工具；rtrim_trailing_zeros 用于移除片段尾部被错误填充的零样本

from __future__ import annotations
import numpy as np


def rtrim_trailing_zeros(y: np.ndarray, floor: float = 0.0, max_strip_samples: int | None = None) -> np.ndarray:
    """
    删除片段尾部的“全零”样本；不动非零淡出尾巴。
    floor=0.0 表示只裁真正的 0；max_strip_samples 可设最大裁剪样本数上限（例如 100ms）。
    """
    if y is None:
        return y
    if y.ndim != 1:
        y = y.reshape(-1)
    n = len(y)
    if n == 0:
        return y

    # 仅在尾部窗口内检查（若提供上限）
    start = 0 if max_strip_samples is None else max(0, n - int(max_strip_samples))

    i = n - 1
    if floor <= 0.0:
        # 只裁掉严格等于 0 的样本
        while i >= start and float(y[i]) == 0.0:
            i -= 1
    else:
        thr = float(floor)
        while i >= start and abs(float(y[i])) <= thr:
            i -= 1

    return y[: i + 1]

