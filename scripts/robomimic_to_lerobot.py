#!/usr/bin/env python3
"""Convert a robomimic image dataset (HDF5) into a LeRobotDataset, for ACT training.

Neither side ships this conversion: LeRobot 0.3.3 has only dataset-version migrations
(`datasets/v2`, `datasets/v21`), and robomimic 0.5 has no LeRobot export. So it lives here.

The emitted schema deliberately mirrors the community ports `ankile/robomimic-ph-*-image`
(key names, dim order, `next.done`, fps), which lets `--verify-against` diff our output
against one of those datasets frame-for-frame. It is also, and more importantly, exactly the
controller's observation contract:

    observation.state              float32 [9]  = eef_pos(3) + eef_quat(4, xyzw) + gripper_qpos(2)
    observation.images.<camera>    uint8 [H,W,3], scene camera FIRST, wrist LAST
    action                         float32 [7]  delta  [pos, axis-angle, gripper], or
                                   float32 [10] abs    [pos, rot_6d, gripper]
    alt.action                     float32 [7]  = the other action convention, when available

Three things here are load-bearing and easy to get wrong:

* **Camera order.** `ACTBackend._obs_to_batch` maps `image_keys[0]` <- agentview and
  `image_keys[1]` <- wrist *positionally*, so the wrist camera is forced last regardless of
  HDF5 ordering. Getting this backwards trains fine and behaves badly in the loop.
* **State layout.** The 9-dim vector must match `obs['proprio']` element for element; the
  backend silently pads/truncates a mismatch rather than raising.
* **`alt.action`'s prefix.** `dataset_to_policy_features` types *any* key starting with
  `action` as a policy output, so the alternate convention cannot be called `action.abs`
  without ACT trying to predict it too. Anything not prefixed `observation`/`action` is
  skipped, hence `alt.`: carried for provenance and mode switching, invisible to training.

Absolute actions (invariant 1 — plant, reactive layer and policy must agree on the mode) come
in two flavours, `--actions`:

  abs          read an `actions_abs` key, as written by robomimic's replay-based converter
               (scripts/conversion/robosuite_add_absolute_actions.py). That converter cannot run
               on the datasets we have: they were collected under robosuite 1.2, whose model XMLs
               no longer load in the pinned 1.4.1 ("No geom with name robot0_g0_vis"), and
               robomimic 0.5 now distributes only v1.5 raw states. Kept for datasets that do
               carry the key.
  abs-derived  compute the targets from the recorded EE pose instead of replaying, which needs no
               simulator at all. Exact under `control_delta=True`: robosuite's OSC sets its goal
               from the *current* pose every control step, so the target is
               `p + output_max[:3] * a[:3]` and `R(output_max[3:] * a[3:6]) @ R_current`. The
               scales come from the dataset's own controller config, and the quaternion algebra is
               `evh_reactive.transforms` — the same code the reactive layer tracks with, so the
               dataset cannot drift from the runtime convention.

Absolute rotation is written as **rot_6d**, making the action 10-dim `[pos, rot_6d, gripper]` —
the same layout as the DP abs-action checkpoints, and what `ACTBackend` recognises and converts
back to the 7-dim contract. This is not a stylistic choice. Measured on these Lift demos, 100%
of the absolute orientation targets sit within 0.12 rad of the pi wrap (the gripper points down
for the whole task), 48% of frames flip sign, and consecutive frames jump by up to 2 pi with no
motion behind them. A policy regressing that axis-angle target is being asked to average two
antipodal encodings of the same rotation; the first ACT trained this way reached 0/3 on Lift with
0.45 rad of rotation error under teacher forcing. `--rot-repr axis_angle` keeps the old 7-dim
output for comparison. Delta actions are unaffected — their rotations are small and continuous.

Usage (needs h5py + lerobot; `.venv-local` has both, as does the host image):

    .venv-local/bin/python scripts/robomimic_to_lerobot.py \
        --dataset data/robomimic/lift/ph/image.hdf5 --out data/lerobot/lift_ph
    # absolute-action variant, then cross-check the delta one against a community port
    ... --actions abs-derived --out data/lerobot/lift_ph_abs
    ... --verify-against data/lerobot/ankile-ph-lift-image
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, 'ros2_ws', 'src', 'evh_reactive'))
sys.path.insert(0, os.path.join(_REPO, 'ros2_ws', 'src', 'evh_controller'))

from evh_controller.rotation import abs7_to_abs10  # noqa: E402
from evh_reactive.transforms import axisangle_to_quat, quat_mul, quat_to_axisangle  # noqa: E402

# Human-readable instruction per robosuite env. ACT is not language-conditioned, so this only
# labels the episodes -- but a dataset that says "Lift" and nothing else is hard to read later.
TASK_INSTRUCTIONS = {
    'Lift': 'lift the cube',
    'PickPlaceCan': 'pick up the can and place it in the bin',
    'NutAssemblySquare': 'place the square nut on the peg',
    'ToolHang': 'assemble the frame and hang the tool',
    'TwoArmTransport': 'transport the payload between the arms',
}

STATE_KEYS = ('robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos')

# robosuite's OSC does not control the frame `robot0_eef_quat` reports: its tool frame is that
# one rotated -90 deg about z (measured exactly, and constant — `R_ctrl = R_obs @ Rz(-pi/2)`,
# positions identical). Absolute ORIENTATION targets are consumed by the controller, so they must
# be expressed in ITS frame; deltas are immune because the controller composes them with its own
# current orientation. Skipping this is silent: the arm is commanded a 90-degree twist it fights
# for the whole episode, and an ACT trained on such targets scored 0/10 on Lift while the same
# demos in delta form scored 7/10. The DP abs checkpoints agree with the corrected convention —
# their targets come from replaying and reading the controller's own goal.
EEF_TO_CONTROL_QUAT = np.array([0.0, 0.0, -np.sin(np.pi / 4), np.cos(np.pi / 4)])

# Written into the dataset's meta/ so the action convention is recoverable after conversion.
ACTION_MODE_FILE = 'evh_action_mode.json'


def sorted_demos(data: dict) -> list[str]:
    """Demo keys in numeric order — h5py yields them alphabetically (demo_10 before demo_2)."""
    return sorted(data.keys(), key=lambda name: int(name.split('_')[1]))


def camera_keys(obs: dict, wrist_hint: str = 'eye_in_hand') -> list[str]:
    """robomimic `*_image` obs keys, scene camera(s) first and the wrist camera last."""
    cams = [k for k in obs.keys() if k.endswith('_image')]
    return sorted(cams, key=lambda k: (wrist_hint in k, k))


def state_names(keys: tuple[str, ...], dims: list[int]) -> list[str]:
    """Per-element names, e.g. robot0_eef_pos_0 — same convention as the community ports."""
    return [f'{key}_{i}' for key, dim in zip(keys, dims) for i in range(dim)]


def osc_scales(env_meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """OSC output_max, i.e. how far a unit action moves the goal — invariant 3's numbers."""
    cfg = env_meta['env_kwargs'].get('controller_configs') or {}
    output_max = cfg.get('output_max')
    if output_max is None:                      # robosuite 1.5 nests it per body part
        for part in cfg.get('body_parts', {}).values():
            if isinstance(part, dict) and str(part.get('type', '')).startswith('OSC'):
                output_max = part.get('output_max')
    if output_max is None:
        raise SystemExit('dataset controller config has no output_max — cannot derive absolute '
                         'actions; pass --actions delta, or use a dataset with actions_abs')
    output_max = np.asarray(output_max, dtype=np.float64)
    return output_max[:3], output_max[3:6]


def derive_absolute(actions: np.ndarray, eef_pos: np.ndarray, eef_quat: np.ndarray,
                    pos_scale: np.ndarray, rot_scale: np.ndarray) -> np.ndarray:
    """Delta OSC actions -> absolute EE targets, using the pose each action was issued from.

    Orientation is composed in the controller's tool frame (see EEF_TO_CONTROL_QUAT), which is
    what an absolute action is read in; position needs no correction (the two frames share an
    origin).
    """
    clipped = np.clip(actions[:, :6], -1.0, 1.0)   # OSC clips to input_min/input_max first
    out = np.empty_like(actions)
    out[:, :3] = eef_pos + clipped[:, :3] * pos_scale
    for t in range(len(actions)):
        current = quat_mul(eef_quat[t], EEF_TO_CONTROL_QUAT)   # into the controller's tool frame
        goal = quat_mul(axisangle_to_quat(clipped[t, 3:6] * rot_scale), current)
        out[t, 3:6] = quat_to_axisangle(goal)
    out[:, 6:] = actions[:, 6:]                    # gripper passes through
    return out


def action_names(action_dim: int) -> list[str]:
    """Name the action dims so a reader can tell the 10-dim abs layout from a 7-dim one."""
    if action_dim == 10:
        return ['pos_x', 'pos_y', 'pos_z', *[f'rot6d_{i}' for i in range(6)], 'gripper']
    if action_dim == 7:
        return ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'gripper']
    return [f'action_{i}' for i in range(action_dim)]


def build_features(cams: list[str], state_dim: int, state_labels: list[str],
                   action_dim: int, image_shape: tuple[int, int, int],
                   use_videos: bool, with_alt: bool, alt_dim: int = 7) -> dict:
    image_dtype = 'video' if use_videos else 'image'
    features = {
        'action': {'dtype': 'float32', 'shape': (action_dim,), 'names': action_names(action_dim)},
        'next.done': {'dtype': 'bool', 'shape': (1,), 'names': ['done']},
        'observation.state': {'dtype': 'float32', 'shape': (state_dim,), 'names': state_labels},
    }
    if with_alt:
        features['alt.action'] = {'dtype': 'float32', 'shape': (alt_dim,), 'names': None}
    for cam in cams:
        features[f'observation.images.{cam[:-len("_image")]}'] = {
            'dtype': image_dtype, 'shape': image_shape, 'names': ['height', 'width', 'channel'],
        }
    return features


def select_demos(f: dict, filter_key: str, limit: int) -> list[str]:
    """Episode list, optionally restricted to one of the file's own train/valid masks."""
    if filter_key:
        if 'mask' not in f or filter_key not in f['mask']:
            available = list(f['mask'].keys()) if 'mask' in f else []
            raise SystemExit(f'no mask/{filter_key} in dataset (available: {available})')
        demos = [name.decode() if isinstance(name, bytes) else str(name)
                 for name in f['mask'][filter_key][:]]
        demos = sorted(demos, key=lambda name: int(name.split('_')[1]))
    else:
        demos = sorted_demos(f['data'])
    return demos[:limit] if limit else demos


def convert(args: argparse.Namespace) -> Path:
    import h5py
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f'{out} exists (pass --overwrite to replace it)')
        shutil.rmtree(out)

    derived = args.actions == 'abs-derived'
    action_key = 'actions' if args.actions == 'delta' else 'actions_abs'
    alt_key = 'actions_abs' if args.actions == 'delta' else 'actions'

    with h5py.File(args.dataset, 'r') as f:
        data = f['data']
        env_meta = json.loads(data.attrs['env_args'])
        env_name = env_meta['env_name']
        fps = int(env_meta['env_kwargs']['control_freq'])
        demos = select_demos(f, args.filter_key, args.max_episodes)

        probe = data[demos[0]]
        pos_scale, rot_scale = osc_scales(env_meta) if derived else (None, None)
        if derived:
            action_key, alt_key = 'actions', 'actions'   # derived from the delta actions
        if action_key not in probe:
            raise SystemExit(
                f"'{action_key}' missing from {args.dataset} — run robomimic's "
                'conversion/robosuite_add_absolute_actions.py first (see this script\'s docstring)')
        with_alt = derived or alt_key in probe
        cams = camera_keys(probe['obs'])
        if not cams:
            raise SystemExit(f'{args.dataset} has no *_image observations — use an image.hdf5, '
                             'or render one with robomimic/scripts/dataset_states_to_obs.py')
        image_shape = tuple(probe['obs'][cams[0]].shape[1:])
        dims = [probe['obs'][k].shape[1] for k in STATE_KEYS]
        labels = state_names(STATE_KEYS, dims)
        raw_action_dim = probe[action_key].shape[1]
        # abs + rot_6d -> the 10-dim [pos, rot_6d, gripper] layout ACTBackend recognises
        rot6d = args.actions != 'delta' and args.rot_repr == 'rot_6d' and raw_action_dim == 7
        action_dim = 10 if rot6d else raw_action_dim
        task = args.task or TASK_INSTRUCTIONS.get(env_name, env_name)

        note = ''
        if derived:
            note = f'  [derived from deltas, output_max={pos_scale[0]:g}/{rot_scale[0]:g}]'
        if rot6d:
            note += '  [rot_6d]'
        elif not with_alt:
            note = f"  [no '{alt_key}' — alt.action omitted]"
        print(f'{args.dataset}: env={env_name} fps={fps} demos={len(demos)} '
              f'cams={cams} state={sum(dims)}d action={action_dim}d ({args.actions}){note}')

        dataset = LeRobotDataset.create(
            repo_id=args.repo_id or f'local/{out.name}',
            fps=fps,
            root=out,
            robot_type='robomimic',
            features=build_features(cams, sum(dims), labels, action_dim, image_shape,
                                    args.images == 'video', with_alt, alt_dim=raw_action_dim),
            use_videos=args.images == 'video',
            image_writer_threads=args.image_writer_threads,
        )

        for i, demo in enumerate(demos):
            ep = data[demo]
            dones = ep['dones'][:].astype(bool)
            state = np.concatenate([ep['obs'][k][:] for k in STATE_KEYS],
                                   axis=1).astype(np.float32)
            if derived:
                raw = ep['actions'][:].astype(np.float64)
                actions = derive_absolute(raw, ep['obs'][STATE_KEYS[0]][:],
                                          ep['obs'][STATE_KEYS[1]][:],
                                          pos_scale, rot_scale).astype(np.float32)
                alt = raw.astype(np.float32)
            else:
                actions = ep[action_key][:].astype(np.float32)
                alt = ep[alt_key][:].astype(np.float32) if with_alt else None
            if rot6d:
                actions = abs7_to_abs10(actions)
            images = {cam: ep['obs'][cam][:] for cam in cams}

            for t in range(len(actions)):
                frame = {
                    'action': actions[t],
                    'next.done': np.array([dones[t]]),
                    'observation.state': state[t],
                }
                if with_alt:
                    frame['alt.action'] = alt[t]
                for cam in cams:
                    frame[f'observation.images.{cam[:-len("_image")]}'] = images[cam][t]
                dataset.add_frame(frame, task=task)
            dataset.save_episode()
            if (i + 1) % 25 == 0 or i + 1 == len(demos):
                print(f'  {i + 1}/{len(demos)} episodes')

    # The action convention travels with the dataset, so the checkpoint trained on it can be
    # stamped from here rather than by hand (scripts/stamp_act_checkpoint.py -> invariant 1).
    mode_path = out / 'meta' / ACTION_MODE_FILE
    mode_path.write_text(json.dumps({'absolute_actions': args.actions != 'delta',
                                     'actions': args.actions,
                                     'rot_repr': args.rot_repr}, indent=2) + '\n')
    print(f'wrote {dataset.meta.total_episodes} episodes / {dataset.meta.total_frames} frames '
          f'-> {out}')
    return out


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean())
    return float('inf') if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def _frame_image(sample: dict, key: str) -> np.ndarray:
    """LeRobot returns CHW float in [0,1]; back to the HWC uint8 scale we compare in."""
    return (sample[key].numpy().transpose(1, 2, 0) * 255.0).round()


def self_check(root: Path, args: argparse.Namespace) -> bool:
    """Read the produced dataset back and diff it against the HDF5 it came from."""
    import h5py
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=f'local/{root.name}', root=root, video_backend='pyav')
    derived = args.actions == 'abs-derived'
    rot6d = args.actions != 'delta' and args.rot_repr == 'rot_6d'
    action_key = 'actions' if args.actions in ('delta', 'abs-derived') else 'actions_abs'
    # `action` holds derived and/or 6D-encoded values with no counterpart in the HDF5; alt.action
    # is the untouched source column, so that is what the alignment check can compare.
    frame_key = 'alt.action' if (derived or rot6d) else 'action'
    episodes = args.check_episodes
    ok = True
    with h5py.File(args.dataset, 'r') as f:
        demos = select_demos(f, args.filter_key, args.max_episodes)   # same order convert() used
        starts, ends = ds.episode_data_index['from'], ds.episode_data_index['to']
        for ep in sorted({0, ds.meta.total_episodes // 2, ds.meta.total_episodes - 1})[:episodes]:
            src = f['data'][demos[ep]]
            i0, i1 = int(starts[ep]), int(ends[ep])
            first, last = ds[i0], ds[i1 - 1]
            n_src, n_out = src[action_key].shape[0], i1 - i0
            a_err = max(float(np.abs(first[frame_key].numpy() - src[action_key][0]).max()),
                        float(np.abs(last[frame_key].numpy() - src[action_key][n_src - 1]).max()))
            s_src = np.concatenate([src['obs'][k][0] for k in STATE_KEYS])
            s_err = float(np.abs(first['observation.state'].numpy() - s_src).max())
            cams = camera_keys(src['obs'])
            img_err = max(
                float(np.abs(_frame_image(first, f'observation.images.{c[:-len("_image")]}')
                             - src['obs'][c][0].astype(np.float32)).max()) for c in cams)
            good = n_src == n_out and a_err < 1e-5 and s_err < 1e-5 and img_err == 0
            ok &= good
            print(f'  {demos[ep]}: len {n_src}=={n_out} action {a_err:.1e} state {s_err:.1e} '
                  f'image maxdiff {img_err:.0f}  {"ok" if good else "MISMATCH"}')
    return ok


def verify_against(root: Path, ref_root: str, episodes: int) -> bool:
    """Cross-check against another LeRobotDataset of the same source (e.g. a community port).

    Numbers must agree; images need not — a video-encoded reference is lossy, so its PSNR is
    reported rather than asserted.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ours = LeRobotDataset(repo_id=f'local/{root.name}', root=root, video_backend='pyav')
    ref = LeRobotDataset(repo_id=f'local/{Path(ref_root).name}', root=ref_root,
                         video_backend='pyav')
    print(f'cross-check vs {ref_root}: {ref.meta.total_episodes} episodes / '
          f'{ref.meta.total_frames} frames, fps {ref.fps}')
    if (ours.meta.total_frames, ours.fps) != (ref.meta.total_frames, ref.fps):
        print('  WARNING: frame count or fps differs — episode indices may not correspond')

    shared_images = sorted({k for k in ours.meta.features if k.startswith('observation.images')}
                           & set(ref.meta.features))
    ok = True
    for ep in sorted({0, ref.meta.total_episodes // 2, ref.meta.total_episodes - 1})[:episodes]:
        i_ours, i_ref = int(ours.episode_data_index['from'][ep]), int(ref.episode_data_index['from'][ep])
        n_ours = int(ours.episode_data_index['to'][ep]) - i_ours
        n_ref = int(ref.episode_data_index['to'][ep]) - i_ref
        a, b = ours[i_ours], ref[i_ref]
        a_err = float(np.abs(a['action'].numpy() - b['action'].numpy()).max())
        s_err = float(np.abs(a['observation.state'].numpy() - b['observation.state'].numpy()).max())
        psnrs = ' '.join(f'{k.split(".")[-1]}={_psnr(_frame_image(a, k), _frame_image(b, k)):.1f}dB'
                         for k in shared_images)
        good = n_ours == n_ref and a_err < 1e-5 and s_err < 1e-5
        ok &= good
        print(f'  episode {ep}: len {n_ours}=={n_ref} action {a_err:.1e} state {s_err:.1e} | '
              f'{psnrs}  {"ok" if good else "MISMATCH"}')
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dataset', required=True, help='robomimic image.hdf5')
    p.add_argument('--out', required=True, help='output LeRobotDataset root')
    p.add_argument('--repo-id', default='', help='dataset id in the metadata (default local/<out>)')
    p.add_argument('--actions', choices=('delta', 'abs', 'abs-derived'), default='delta',
                   help="abs reads an 'actions_abs' key; abs-derived computes the targets from "
                        'the recorded EE pose (no simulator needed) — see the module docstring')
    p.add_argument('--rot-repr', choices=('rot_6d', 'axis_angle'), default='rot_6d',
                   help='absolute rotation encoding; rot_6d makes the action 10-dim and is the '
                        'only continuous one where these targets live (see the docstring)')
    p.add_argument('--images', choices=('image', 'video'), default='image',
                   help='image = lossless PNG; video is ~40x smaller but visibly lossy at 84x84')
    p.add_argument('--filter-key', default='', help="dataset mask to use, e.g. train / valid")
    p.add_argument('--max-episodes', type=int, default=0)
    p.add_argument('--task', default='', help='instruction string (default: per-env sentence)')
    p.add_argument('--image-writer-threads', type=int, default=8)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--check-episodes', type=int, default=3,
                   help='episodes sampled for the post-conversion self-check (0 to skip)')
    p.add_argument('--verify-against', default='',
                   help='root of another LeRobotDataset of the same demos to diff against')
    p.add_argument('--verify-only', action='store_true',
                   help='skip conversion; only run the checks against an existing --out')
    args = p.parse_args()

    out = Path(args.out) if args.verify_only else convert(args)

    ok = True
    if args.check_episodes:
        print('self-check against the source HDF5:')
        ok &= self_check(out, args)
    if args.verify_against:
        ok &= verify_against(out, args.verify_against, args.check_episodes or 3)
    if not ok:
        sys.exit('verification FAILED')


if __name__ == '__main__':
    main()
