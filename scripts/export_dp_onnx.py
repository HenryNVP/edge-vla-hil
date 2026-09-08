#!/usr/bin/env python3
"""Export a diffusion_policy-repo image checkpoint to ONNX for the Jetson controller.

Unlike ACT (`export_onnx.py`), a diffusion policy is not one forward pass — it is an observation
encoder plus `num_inference_steps` passes through a UNet, with a DDIM update between them. This
exporter writes **two graphs and a sidecar**, and leaves the loop in Python:

    <out>/encoder.onnx   images + proprio  ->  global_cond [1, To*Do]
    <out>/unet.onnx      (sample, timestep, global_cond) -> epsilon [1, T, Da]
    <out>/meta.json      shapes, the scheduler's own constants, the action normalizer

Two reasons not to unroll the loop into a single graph, which is the obvious alternative:

  * size. The UNet is 256M params against the encoder's 22M; unrolling bakes the step count in
    and gives the Jetson a graph it has no reason to hold.
  * RTC. `predict_inpaint` applies its guidance *between* denoising steps (see
    dp_repo_policy.predict_inpaint). With the loop inside the graph there is nowhere to put it,
    and the ONNX backend would silently fall back to the soft-blend that RTC exists to replace.

Normalization is baked into the encoder graph, so the Jetson feeds raw float images in [0, 1] and
raw proprio. Action unnormalization is a linear map, so its two vectors ride in the sidecar.

The DDIM constants are read off the checkpoint's own scheduler rather than re-derived from a beta
schedule — and `--check` re-runs the whole sampler against torch with shared initial noise, which
is the only honest way to know the numpy loop in `dp_onnx_policy.py` agrees with diffusers.

    python scripts/export_dp_onnx.py --ckpt checkpoints/dp_lift_ph_image_cnn.ckpt \
        --out outputs/dp_lift_onnx --steps 16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / 'ros2_ws' / 'src' / 'evh_controller'))

from evh_controller.dp_repo_policy import load_dp_checkpoint  # noqa: E402

# checkpoint obs keys, in the order the exported graph takes them
IMAGE_KEYS = ('agentview_image', 'robot0_eye_in_hand_image')
LOWDIM_KEYS = ('robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos')
OBS_KEYS = IMAGE_KEYS + LOWDIM_KEYS
INPUT_NAMES = ('agentview', 'wrist', 'eef_pos', 'eef_quat', 'gripper')


class EncoderGraph(nn.Module):
    """Normalize, encode, flatten: obs tensors -> global_cond [1, To*Do].

    The normalizer is inlined as buffers rather than called through `LinearNormalizer`, whose
    dict-in/dict-out signature does not trace. `_norm` reproduces its arithmetic exactly
    (`x.reshape(-1, scale.shape[0]) * scale + offset`); `--check` proves it did.
    """

    def __init__(self, policy) -> None:
        super().__init__()
        self.obs_encoder = policy.obs_encoder
        for key in OBS_KEYS:
            params = policy.normalizer.params_dict[key]
            self.register_buffer(f'{key}__scale', params['scale'].detach().clone())
            self.register_buffer(f'{key}__offset', params['offset'].detach().clone())

    def _norm(self, x: torch.Tensor, key: str) -> torch.Tensor:
        scale = getattr(self, f'{key}__scale')
        offset = getattr(self, f'{key}__offset')
        return (x.reshape(-1, scale.shape[0]) * scale + offset).reshape(x.shape)

    def forward(self, agentview, wrist, eef_pos, eef_quat, gripper):
        tensors = (agentview, wrist, eef_pos, eef_quat, gripper)
        obs = {key: self._norm(t, key) for key, t in zip(OBS_KEYS, tensors)}
        return self.obs_encoder(obs).reshape(1, -1)


def freeze_center_crops(module: nn.Module) -> int:
    """Replace each CropRandomizer's eval-time centre crop with a constant slice.

    `torchvision.center_crop` computes its offsets with `int(round(...))` over the image height,
    which under tracing is a Tensor and raises `type Tensor doesn't define __round__`. The eval
    path is deterministic — fixed image size, fixed crop size — so the offsets are constants and
    the slice is exactly the same pixels. `main()` proves that by comparing encoder output against
    the unpatched policy before exporting.

    Patched per-instance, not in the vendored repo, which is read-only (see CLAUDE.md).
    """
    patched = 0
    for sub in module.modules():
        if type(sub).__name__ != 'CropRandomizer':
            continue
        _, height, width = sub.input_shape
        ch, cw = int(sub.crop_height), int(sub.crop_width)
        top = int(round((height - ch) / 2.0))
        left = int(round((width - cw) / 2.0))
        crops = int(sub.num_crops)

        def forward_in(inputs, top=top, left=left, ch=ch, cw=cw, crops=crops):
            out = inputs[..., top:top + ch, left:left + cw]
            if crops > 1:
                out = out.unsqueeze(1).expand(-1, crops, -1, -1, -1).reshape(-1, out.shape[-3],
                                                                            ch, cw)
            return out

        sub.forward_in = forward_in
        patched += 1
    return patched


class UNetGraph(nn.Module):
    """One denoising step: (sample, timestep, global_cond) -> predicted noise."""

    def __init__(self, policy) -> None:
        super().__init__()
        self.model = policy.model

    def forward(self, sample, timestep, global_cond):
        return self.model(sample, timestep, local_cond=None, global_cond=global_cond)


def _dummy_inputs(policy, device):
    """One batch of the encoder's inputs, in the graph's units (images already in [0, 1])."""
    to = int(policy.n_obs_steps)
    c, h, w = policy.obs_encoder.key_shape_map['agentview_image'] \
        if hasattr(policy.obs_encoder, 'key_shape_map') else (3, 84, 84)
    return (
        torch.rand(to, c, h, w, device=device),
        torch.rand(to, c, h, w, device=device),
        torch.rand(to, 3, device=device),
        torch.rand(to, 4, device=device),
        torch.rand(to, 2, device=device),
    )


def scheduler_constants(policy, steps: int) -> dict:
    """The sampler's own numbers, so the numpy loop cannot drift from a re-derived schedule."""
    sched = policy.noise_scheduler
    sched.set_timesteps(steps)
    cfg = sched.config
    return {
        'num_inference_steps': int(steps),
        'num_train_timesteps': int(cfg.num_train_timesteps),
        'timesteps': [int(t) for t in sched.timesteps],
        'alphas_cumprod': [float(a) for a in sched.alphas_cumprod],
        'final_alpha_cumprod': float(sched.final_alpha_cumprod),
        'prediction_type': str(cfg.prediction_type),
        'clip_sample': bool(cfg.clip_sample),
        'clip_sample_range': float(getattr(cfg, 'clip_sample_range', 1.0)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', default='checkpoints/dp_lift_ph_image_cnn.ckpt')
    p.add_argument('--out', default='outputs/dp_lift_onnx',
                   help='directory for encoder.onnx, unet.onnx and meta.json')
    p.add_argument('--steps', type=int, default=16, help='DDIM inference steps to bake in')
    p.add_argument('--opset', type=int, default=17)
    p.add_argument('--device', default='cuda', help='export device; cpu works and is slower')
    p.add_argument('--check', action='store_true',
                   help='after exporting, re-run the full sampler through ONNX Runtime and '
                        'compare with torch on shared initial noise')
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    policy, cfg = load_dp_checkpoint(args.ckpt, args.device, args.steps)
    policy.eval()
    device = policy.device

    if not getattr(policy, 'obs_as_global_cond', False):
        raise SystemExit(
            'this checkpoint conditions by obs-inpainting, not a global condition vector — the '
            'two-graph split assumes the latter (see the module docstring)')

    to = int(policy.n_obs_steps)
    horizon = int(policy.horizon)
    raw_action_dim = int(policy.action_dim)
    dummies = _dummy_inputs(policy, device)

    # --- encoder ---------------------------------------------------------------
    encoder = EncoderGraph(policy).to(device).eval()
    with torch.no_grad():
        # reference FIRST, with the checkpoint's own crop and normalizer untouched
        reference = policy.obs_encoder(policy.normalizer.normalize(
            dict(zip(OBS_KEYS, dummies)))).reshape(1, -1)
    crops = freeze_center_crops(encoder)
    with torch.no_grad():
        global_cond = encoder(*dummies)
    drift = float((global_cond - reference).abs().max())
    if drift > 1e-4:
        raise SystemExit(
            f'the traceable encoder disagrees with the checkpoint by {drift:.2e} — the inlined '
            'normalizer or the frozen crop is wrong')
    print(f'[export] encoder matches the policy path (max diff {drift:.2e}, '
          f'{crops} centre crop(s) frozen)')

    torch.onnx.export(
        encoder, dummies, str(out / 'encoder.onnx'), opset_version=args.opset,
        input_names=list(INPUT_NAMES), output_names=['global_cond'], do_constant_folding=True)
    print(f'[export] encoder.onnx  global_cond{tuple(global_cond.shape)}')

    # --- unet ------------------------------------------------------------------
    unet = UNetGraph(policy).to(device).eval()
    sample = torch.randn(1, horizon, raw_action_dim, device=device)
    timestep = torch.tensor([0], dtype=torch.int64, device=device)
    torch.onnx.export(
        unet, (sample, timestep, global_cond), str(out / 'unet.onnx'),
        opset_version=args.opset, input_names=['sample', 'timestep', 'global_cond'],
        output_names=['noise_pred'], do_constant_folding=True)
    print(f'[export] unet.onnx     sample{tuple(sample.shape)} '
          f'global_cond{tuple(global_cond.shape)}')

    # --- sidecar ---------------------------------------------------------------
    action = policy.normalizer.params_dict['action']
    image_shape = list(cfg.shape_meta.obs.agentview_image.shape)
    meta = {
        'kind': 'diffusion_policy_onnx',
        'source_checkpoint': str(Path(args.ckpt).name),
        'encoder': 'encoder.onnx',
        'unet': 'unet.onnx',
        'n_obs_steps': to,
        'horizon': horizon,
        'raw_action_dim': raw_action_dim,
        # the plant's contract is 7-dim; a 10-dim head is the abs [pos, rot_6d, gripper] variant
        'action_dim': 7 if raw_action_dim == 10 else raw_action_dim,
        'absolute_actions': raw_action_dim == 10,
        'chunk_size': horizon - (to - 1),
        'image_shape': image_shape,
        'obs_feature_dim': int(policy.obs_feature_dim),
        'global_cond_dim': int(global_cond.shape[1]),
        'action_scale': [float(v) for v in action['scale'].detach().cpu().numpy()],
        'action_offset': [float(v) for v in action['offset'].detach().cpu().numpy()],
        **scheduler_constants(policy, args.steps),
    }
    (out / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'[export] meta.json     chunk_size={meta["chunk_size"]} '
          f'absolute={meta["absolute_actions"]} steps={args.steps}')

    if args.check:
        _check(policy, out, dummies, meta)


def _check(policy, out: Path, dummies, meta: dict) -> None:
    """Run the exported sampler and the torch one from the SAME initial noise, and compare.

    Anything less proves nothing: the sampler starts from randn, so two correct implementations
    disagree completely on independent draws.
    """
    sys.path.insert(0, str(_REPO / 'ros2_ws' / 'src' / 'evh_controller'))
    import onnxruntime as ort

    from evh_controller.dp_onnx_policy import ddim_sample

    noise = np.random.RandomState(0).randn(
        1, meta['horizon'], meta['raw_action_dim']).astype(np.float32)

    # --- torch reference: conditional_sample with the noise pinned ---
    sched = policy.noise_scheduler
    sched.set_timesteps(meta['num_inference_steps'])
    with torch.no_grad():
        global_cond = policy.obs_encoder(policy.normalizer.normalize(
            dict(zip(OBS_KEYS, dummies)))).reshape(1, -1)
        traj = torch.from_numpy(noise).to(policy.device, policy.dtype)
        for t in sched.timesteps:
            eps = policy.model(traj, t, local_cond=None, global_cond=global_cond)
            traj = sched.step(eps, t, traj).prev_sample
        torch_action = policy.normalizer['action'].unnormalize(traj)[0].cpu().numpy()

    # --- ONNX: same noise, numpy DDIM ---
    providers = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider')
                 if p in ort.get_available_providers()]
    enc = ort.InferenceSession(str(out / 'encoder.onnx'), providers=providers)
    net = ort.InferenceSession(str(out / 'unet.onnx'), providers=providers)
    feeds = {name: t.detach().cpu().numpy() for name, t in zip(INPUT_NAMES, dummies)}
    cond = enc.run(['global_cond'], feeds)[0]

    cond_drift = float(np.abs(cond - global_cond.cpu().numpy()).max())
    traj = ddim_sample(
        noise, cond, meta,
        lambda s, t, c: net.run(['noise_pred'], {
            'sample': s, 'timestep': np.asarray([t], dtype=np.int64), 'global_cond': c})[0])
    onnx_action = (traj[0] - np.asarray(meta['action_offset'], np.float32)) \
        / np.asarray(meta['action_scale'], np.float32)

    action_drift = float(np.abs(onnx_action - torch_action).max())
    scale = float(np.abs(torch_action).max())
    print(f'[check] global_cond max |diff| = {cond_drift:.3e}')
    print(f'[check] action      max |diff| = {action_drift:.3e}  '
          f'({100 * action_drift / max(scale, 1e-9):.3f}% of full scale)')
    if action_drift > 1e-2:
        raise SystemExit('exported sampler does not match torch — do NOT deploy this')
    print('[check] OK')


if __name__ == '__main__':
    main()
