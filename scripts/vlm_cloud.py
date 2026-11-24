"""Frozen VLM cloud for the BEV planner: SmolVLM2-500M reads the instruction, then the image, and
its LM hidden states at the image tokens (layer 16, 8x8 per view, 960 dims) are the cloud's
feature maps. bev_model.CloudBEV(vlm_dim=960) projects them onto the table grid exactly as it
does ResNet maps; its learned 1x1 projection is the only compression.

The instruction goes BEFORE the image: the LM is causal, so image tokens only see earlier text.
Measured on nut_goal (outputs/vlm/probe.py, linear probe on the grid-sampled features, random
instruction per frame so the arm's motion cannot leak it): layer 16 marks the instructed nut's
cell 90% of the time (other nut 6%), SigLIP or ResNet features alone 47-50%; nut localisation is
level with ResNet-18 (98-99% within one cell). Layers 24/32 ground slightly worse.

No fixed compression: a 128-dim PCA kept 92% of the variance but the instructed-nut choice fell
from 0.93 to 0.58, and a basis spanning the instruction difference only recovered 0.72
(outputs/vlm/probe_enc.py); the scene dominates the variance, the instruction is a small shift.

Frames are stored upside down (OpenGL); the VLM sees them upright and the maps are flipped back,
so the sim camera matrices still apply.

    python3 scripts/vlm_cloud.py cache --data data/oracle_nut_goal/npz --out data/oracle_nut_goal/vlm

The cache holds maps [T, view, 960, 8, 8] fp16 per episode (.npy, memory-mapped in training),
under the instruction in force at each frame: a training memory comes from a frame whose goal is
the target's (bev_planner.goal_consistent_t0), so the other instruction is never read.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

NAME = 'HuggingFaceTB/SmolVLM2-500M-Instruct'
LAYER = 16
SIZE = 512                       # one 512 px tile, no splitting: 64 image tokens (8x8)
INSTRUCTIONS = ('Pick up the square nut.', 'Pick up the round nut.')   # nut_goal.NUTS order


class VLMEncoder:
    def __init__(self, device: str = 'cuda') -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.dev = device if torch.cuda.is_available() else 'cpu'
        self.proc = AutoProcessor.from_pretrained(NAME)
        model = AutoModelForImageTextToText.from_pretrained(NAME, dtype=torch.bfloat16)
        lm = model.model.text_model
        lm.layers = lm.layers[:LAYER + 1]      # hidden_states[LAYER] is then pre-norm, as probed
        self.model = model.model.to(self.dev).eval()
        self.img_id = model.config.image_token_id
        self._prompt = {}

    def _ids(self, goal: int):
        """The instruction + image-placeholder token ids, built once per goal (fixed prompt)."""
        if goal not in self._prompt:
            from PIL import Image
            text = self.proc.apply_chat_template(
                [{'role': 'user', 'content': [{'type': 'text', 'text': INSTRUCTIONS[goal]},
                                              {'type': 'image'}]}], add_generation_prompt=False)
            enc = self.proc(text=[text], images=[[Image.new('RGB', (SIZE, SIZE))]],
                            return_tensors='pt', do_image_splitting=False)
            self._prompt[goal] = enc['input_ids'].to(self.dev)
        return self._prompt[goal]

    def maps(self, views: list[np.ndarray], goals) -> 'torch.Tensor':
        """views: per view uint8 [B, H, W, 3] (stored orientation); goals [B] ints ->
        [B, V, 960, 8, 8] float32. Images are resized and normalised on the GPU (the
        processor's own [-1, 1] scaling) instead of through PIL: same input for the cache and
        the live cloud, ~6x faster."""
        import torch
        import torch.nn.functional as F
        goals = np.asarray(goals, int)
        res = []
        for g in np.unique(goals):
            idx = np.where(goals == g)[0]
            ids = self._ids(int(g)).expand(len(idx), -1)
            per_view = []
            for v in views:
                x = torch.from_numpy(np.ascontiguousarray(np.asarray(v)[idx])).to(self.dev)
                x = x.flip(1).permute(0, 3, 1, 2).float() / 255          # upright for the VLM
                x = F.interpolate(x, size=(SIZE, SIZE), mode='bilinear', align_corners=False)
                pv = ((x - 0.5) / 0.5).to(torch.bfloat16)[:, None]
                with torch.no_grad():
                    hs = self.model(input_ids=ids, pixel_values=pv,
                                    output_hidden_states=True).hidden_states[LAYER]
                h = hs[ids == self.img_id].float().reshape(len(idx), 64, -1)
                per_view.append(h.transpose(1, 2).reshape(len(idx), -1, 8, 8).flip(2))
            res.append((idx, torch.stack(per_view, 1)))
        out = torch.empty(len(goals), *res[0][1].shape[1:], device=self.dev)
        for idx, m in res:
            out[torch.from_numpy(idx).to(self.dev)] = m
        return out


def cache(args) -> None:
    files = sorted(glob.glob(os.path.join(args.data, '*.npz')))
    os.makedirs(args.out, exist_ok=True)
    enc = VLMEncoder()
    for n, f in enumerate(files):
        dst = os.path.join(args.out, os.path.basename(f).replace('.npz', '.npy'))
        if os.path.exists(dst):
            continue
        z = np.load(f)
        chunks = []
        for i in range(0, len(z['action']), args.batch):
            sl = slice(i, i + args.batch)
            chunks.append(enc.maps([z['agentview'][sl], z['top'][sl]],
                                   z['goal'][sl]).half().cpu().numpy())
        np.save(dst + '.tmp.npy', np.concatenate(chunks))             # [T, view, 960, 8, 8]
        os.replace(dst + '.tmp.npy', dst)
        print(f'{n + 1}/{len(files)} {os.path.basename(f)}', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('cache')
    c.add_argument('--data', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--batch', type=int, default=32)
    args = ap.parse_args()
    cache(args)


if __name__ == '__main__':
    main()
