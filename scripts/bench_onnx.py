#!/usr/bin/env python3
"""Benchmark an ACT ONNX model with ONNX Runtime (CUDA on Jetson, CPU fallback).

No PyTorch or LeRobot required at inference time — only numpy + onnxruntime.

Host (x86, PyPI GPU wheel):
    python scripts/bench_onnx.py outputs/act_pusht.onnx

Jetson (GPU wheel, NOT plain PyPI — that wheel is CPU-only): the pypi.jetson-ai-lab.io index no
longer serves JetPack 5 builds (jp6 only as of 2026-09), so Dockerfile.jetson pulls the CUDA wheel
out of dustynv/onnxruntime:r35.4.1 instead — see that file's comments. Once built:
    python3 scripts/bench_onnx.py /ws/outputs/act_pusht.onnx --providers CUDAExecutionProvider

Verify providers:
    python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _load_meta(onnx_path: Path) -> dict:
    meta_path = onnx_path.with_suffix('.json')
    if meta_path.is_file():
        return json.loads(meta_path.read_text())
    return {}


def _zeros_for_session(sess) -> dict[str, np.ndarray]:
    feeds: dict[str, np.ndarray] = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        feeds[inp.name] = np.zeros(shape, dtype=np.float32)
    return feeds


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark ACT ONNX with ONNX Runtime.')
    parser.add_argument('onnx', help='path to .onnx model')
    parser.add_argument('--providers', default='',
                        help='comma-separated ORT providers (default: CUDA,CPU)')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iters', type=int, default=20)
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(
            'onnxruntime is not installed.\n'
            '  Host:  pip install onnxruntime-gpu\n'
            '  Jetson: pip install onnxruntime-gpu '
            '--extra-index-url https://pypi.jetson-ai-lab.io/jp5/cu118'
        ) from exc

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise SystemExit(f'model not found: {onnx_path}')

    providers = [p.strip() for p in args.providers.split(',') if p.strip()]
    if not providers:
        available = ort.get_available_providers()
        providers = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider') if p in available]
        if not providers:
            providers = available[:1]

    print(f'loading {onnx_path}')
    print(f'providers: {providers}')
    print(f'available: {ort.get_available_providers()}')

    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    active = sess.get_providers()
    print(f'active:    {active}')

    meta = _load_meta(onnx_path)
    if meta:
        print(f'chunk {meta.get("chunk_size")} action_dim {meta.get("action_dim")} '
              f'image {meta.get("image_shape")} state_dim {meta.get("state_dim")}')

    feeds = _zeros_for_session(sess)
    out_name = sess.get_outputs()[0].name

    for _ in range(args.warmup):
        sess.run([out_name], feeds)

    times: list[float] = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        sess.run([out_name], feeds)
        times.append((time.perf_counter() - t0) * 1e3)

    mean = sum(times) / len(times)
    print(f'inference mean {mean:.1f} ms  min {min(times):.1f}  max {max(times):.1f}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
