#!/usr/bin/env python3
"""Benchmark an ONNX policy with ONNX Runtime (CUDA on Jetson, CPU fallback).

Takes either shape of export:
  * a single ACT graph  -> `outputs/act_pusht.onnx`
  * a Diffusion Policy export DIRECTORY (encoder.onnx + unet.onnx + meta.json, written by
    scripts/export_dp_onnx.py) -> `outputs/dp_lift_onnx`

For DP the number that matters is one full ACTION, not one graph call: a prediction is the encoder
once plus `num_inference_steps` passes through the UNet, so the per-graph timings are reported
alongside the total but the total is the control-loop budget. It runs the real numpy DDIM loop from
`dp_onnx_policy.py`, not a stand-in, so the measured cost includes the scheduler arithmetic.

No PyTorch, LeRobot or diffusion_policy required at inference time — only numpy + onnxruntime,
which is the whole point on a Python 3.8 Jetson image.

Host (x86, PyPI GPU wheel):
    python scripts/bench_onnx.py outputs/act_pusht.onnx
    python scripts/bench_onnx.py outputs/dp_lift_onnx

Jetson (GPU wheel, NOT plain PyPI — that wheel is CPU-only): the pypi.jetson-ai-lab.io index no
longer serves JetPack 5 builds (jp6 only as of 2026-09), so Dockerfile.jetson pulls the CUDA wheel
out of dustynv/onnxruntime:r35.4.1 instead — see that file's comments. Once built:
    python3 scripts/bench_onnx.py /ws/outputs/act_pusht.onnx --providers CUDAExecutionProvider

Verify providers:
    python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"

The HOST image ships CPU-only `onnxruntime` on purpose (the ONNX path exists for the Jetson), so a
GPU number there needs a temporary install plus torch's bundled CUDA libs on the loader path —
without them the CUDA provider is listed as available and then silently falls back to CPU, which
is a ~3x error in the wrong direction:

    pip install onnxruntime-gpu nvidia-cudnn-cu12
    export LD_LIBRARY_PATH=$(python3 -c "import os,nvidia; d=os.path.dirname(nvidia.__file__); \
      print(':'.join(os.path.join(d,x,'lib') for x in os.listdir(d) \
      if os.path.isdir(os.path.join(d,x,'lib'))))"):$LD_LIBRARY_PATH

Always read the `active:` line, not `available:`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / 'ros2_ws' / 'src' / 'evh_controller'))


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


def _bench_diffusion(root: Path, providers, warmup: int, iters: int, ort) -> None:
    """Time a whole DP prediction: encoder once, then the DDIM loop over the UNet."""
    from evh_controller.dp_onnx_policy import ddim_sample

    meta = json.loads((root / 'meta.json').read_text())
    enc = ort.InferenceSession(str(root / meta['encoder']), providers=providers)
    net = ort.InferenceSession(str(root / meta['unet']), providers=providers)
    print(f'active:    {net.get_providers()}')
    print(f'chunk {meta["chunk_size"]} action_dim {meta["action_dim"]} '
          f'n_obs {meta["n_obs_steps"]} steps {meta["num_inference_steps"]} '
          f'image {meta["image_shape"]} absolute {meta["absolute_actions"]}')

    feeds = _zeros_for_session(enc)
    horizon, raw_dim = int(meta['horizon']), int(meta['raw_action_dim'])
    unet_ms: list[float] = []

    def denoise(sample, timestep, cond):
        t0 = time.perf_counter()
        out = net.run(['noise_pred'], {
            'sample': sample, 'timestep': np.asarray([timestep], dtype=np.int64),
            'global_cond': cond})[0]
        unet_ms.append((time.perf_counter() - t0) * 1e3)
        return out

    def one_action():
        t0 = time.perf_counter()
        cond = enc.run(['global_cond'], feeds)[0]
        t_enc = (time.perf_counter() - t0) * 1e3
        noise = np.random.randn(1, horizon, raw_dim).astype(np.float32)
        ddim_sample(noise, cond, meta, denoise)
        return t_enc, (time.perf_counter() - t0) * 1e3

    for _ in range(warmup):
        one_action()
    unet_ms.clear()

    enc_ms, total_ms = [], []
    for _ in range(iters):
        e, t = one_action()
        enc_ms.append(e)
        total_ms.append(t)

    steps = int(meta['num_inference_steps'])
    print(f'encoder    mean {np.mean(enc_ms):.1f} ms')
    print(f'unet step  mean {np.mean(unet_ms):.1f} ms  x{steps} steps '
          f'= {np.mean(unet_ms) * steps:.1f} ms')
    print(f'ACTION     mean {np.mean(total_ms):.1f} ms  min {min(total_ms):.1f}  '
          f'max {max(total_ms):.1f}  p95 {np.percentile(total_ms, 95):.1f}')
    budget = 1000.0 / 20.0
    verdict = 'fits' if np.mean(total_ms) <= budget else 'does NOT fit'
    print(f'           {verdict} the 20 Hz control budget ({budget:.0f} ms)')


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark an ONNX policy with ONNX Runtime.')
    parser.add_argument('onnx', help='path to an ACT .onnx model, or a DP export directory')
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
    if not onnx_path.exists():
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

    if onnx_path.is_dir():
        _bench_diffusion(onnx_path, providers, args.warmup, args.iters, ort)
        return

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
