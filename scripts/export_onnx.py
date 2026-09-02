#!/usr/bin/env python3
"""Export a LeRobot ACT checkpoint to ONNX (host / Python 3.10+).

The graph is a single-camera inference path: float image + state in, action chunk out.
Normalization and unnormalization are baked in. Run on x86 or any machine with
``pip install lerobot==0.3.3`` — not on the Jetson controller image (Python 3.8).

    python scripts/export_onnx.py --repo lerobot/act_aloha_sim_transfer_cube_human \
        --out outputs/act_aloha.onnx

Then sanity-check on the host:

    python scripts/bench_onnx.py outputs/act_aloha.onnx

On Jetson see docker/Dockerfile.jetson for how the GPU onnxruntime wheel gets installed
(the Jetson AI Lab pip index dropped JetPack 5 support; bench_onnx.py has the detail).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn


def _import_act_policy():
    tried: list[str] = []
    for modpath in (
        'lerobot.policies.act.modeling_act',
        'lerobot.common.policies.act.modeling_act',
    ):
        try:
            return __import__(modpath, fromlist=['ACTPolicy']).ACTPolicy
        except Exception as exc:
            tried.append(f'  {modpath}: {type(exc).__name__}: {exc}')
    raise ImportError(
        'Could not import LeRobot ACTPolicy. Tried:\n'
        + '\n'.join(tried)
        + '\nInstall lerobot==0.3.3 (Python 3.10+).'
    )


def _state_key(policy) -> str:
    key = 'observation.state'
    if key in policy.config.input_features:
        return key
    for k, feat in policy.config.input_features.items():
        feat_type = getattr(feat, 'type', None)
        if feat_type is not None and str(feat_type).endswith('STATE'):
            return k
        if 'state' in k:
            return k
    return key


class ACTOnnxWrapper(nn.Module):
    """Trace-friendly single-camera ACT inference (no Python dict/list in the graph)."""

    def __init__(self, policy) -> None:
        super().__init__()
        self.normalize_inputs = policy.normalize_inputs
        self.unnormalize_outputs = policy.unnormalize_outputs
        self.model = policy.model
        self.image_key = list(policy.config.image_features.keys())[0]
        self.state_key = _state_key(policy)
        self.latent_dim = int(policy.config.latent_dim)
        self.chunk_size = int(policy.config.chunk_size)

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        import einops

        batch = self.normalize_inputs({self.image_key: image, self.state_key: state})
        image_n = batch[self.image_key]
        state_n = batch[self.state_key]
        model = self.model
        batch_size = image_n.shape[0]

        latent_sample = torch.zeros(
            (batch_size, self.latent_dim), device=image_n.device, dtype=image_n.dtype,
        )
        encoder_in_tokens = [model.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(model.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if model.config.robot_state_feature:
            encoder_in_tokens.append(model.encoder_robot_state_input_proj(state_n))

        cam_features = model.backbone(image_n)['feature_map']
        cam_pos_embed = model.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
        cam_features = model.encoder_img_feat_input_proj(cam_features)
        cam_features = einops.rearrange(cam_features, 'b c h w -> (h w) b c')
        cam_pos_embed = einops.rearrange(cam_pos_embed, 'b c h w -> (h w) b c')
        encoder_in_tokens.extend(torch.unbind(cam_features, dim=0))
        encoder_in_pos_embed.extend(torch.unbind(cam_pos_embed, dim=0))

        encoder_in_tokens = torch.stack(encoder_in_tokens, dim=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, dim=0)
        encoder_out = model.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        decoder_in = torch.zeros(
            (self.chunk_size, batch_size, model.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = model.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=model.decoder_pos_embed.weight.unsqueeze(1),
        )
        actions = model.action_head(decoder_out.transpose(0, 1))
        return self.unnormalize_outputs({'action': actions})['action']


def _dummy_inputs(policy, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    image_key = list(policy.config.image_features.keys())[0]
    state_key = _state_key(policy)
    c, h, w = policy.config.input_features[image_key].shape
    state_dim = policy.config.input_features[state_key].shape[0]
    image = torch.zeros(1, c, h, w, device=device)
    state = torch.zeros(1, state_dim, device=device)
    return image, state


def main() -> None:
    parser = argparse.ArgumentParser(description='Export LeRobot ACT to ONNX.')
    parser.add_argument('--repo', default='lerobot/act_aloha_sim_transfer_cube_human',
                        help='HuggingFace repo id or local pretrained dir')
    parser.add_argument('--out', default='outputs/act.onnx')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--device', default='cpu',
                        help='export device (cpu is fine; weights are the same)')
    args = parser.parse_args()

    ACTPolicy = _import_act_policy()
    print(f'loading {args.repo}')
    policy = ACTPolicy.from_pretrained(args.repo)
    policy.eval()

    wrapper = ACTOnnxWrapper(policy).eval().to(args.device)
    image, state = _dummy_inputs(policy, args.device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (image, state),
            str(out_path),
            input_names=['image', 'state'],
            output_names=['action_chunk'],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=None,
        )

    image_key = list(policy.config.input_features.keys())[0]
    state_key = _state_key(policy)
    meta = {
        'repo': args.repo,
        'image_key': image_key,
        'state_key': state_key,
        'image_shape': list(policy.config.input_features[image_key].shape),
        'state_dim': int(policy.config.input_features[state_key].shape[0]),
        'chunk_size': int(policy.config.chunk_size),
        'action_dim': int(policy.config.action_feature.shape[0]),
        'opset': args.opset,
    }
    meta_path = out_path.with_suffix('.json')
    meta_path.write_text(json.dumps(meta, indent=2) + '\n')
    print(f'wrote {out_path}  chunk={meta["chunk_size"]} action_dim={meta["action_dim"]}')
    print(f'wrote {meta_path}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
