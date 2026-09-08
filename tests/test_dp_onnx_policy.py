"""Tests for dp_onnx_policy.py — the Jetson's diffusion path.

`scripts/export_dp_onnx.py --check` already proves the two GRAPHS match torch. What it cannot
prove is the plumbing around them, which is where this backend can silently disagree with
`dp_repo_policy`: history padding, the image resize, which proprio slice becomes which input,
the action unnormalize, and the 10-dim -> 7-dim conversion. A mismatch there produces a policy
that runs at full speed and drives the arm somewhere else.

The DDIM loop is tested against closed-form values rather than against a second implementation
of itself, and `test_ddim_matches_the_torch_scheduler` (integration, needs the checkpoint) is the
one that pins it to diffusers.
"""
import numpy as np
import pytest
from conftest import requires_ros2

META = {
    'kind': 'diffusion_policy_onnx',
    'num_inference_steps': 4,
    'num_train_timesteps': 100,
    'timesteps': [75, 50, 25, 0],
    'alphas_cumprod': list(np.linspace(0.999, 0.01, 100)),
    'final_alpha_cumprod': 1.0,
    'prediction_type': 'epsilon',
    'clip_sample': True,
    'clip_sample_range': 1.0,
}


def _stub_backend(**overrides):
    """A DiffusionONNXBackend with the sessions stubbed out — obs plumbing only."""
    import types

    from evh_controller.dp_onnx_policy import DiffusionONNXBackend

    b = DiffusionONNXBackend.__new__(DiffusionONNXBackend)
    b.n_obs_steps = 2
    b.chunk_size = 15
    b.action_dim = 7
    b.absolute_actions = True
    b._horizon = 16
    b._raw_action_dim = 10
    b._image_hw = (84, 84)
    b._action_scale = np.ones(10, np.float32)
    b._action_offset = np.zeros(10, np.float32)
    b._meta = META
    b._enc = b._unet = types.SimpleNamespace()
    b.__dict__.update(overrides)
    return b


def _obs(n_frames=2, size=84):
    rng = np.random.RandomState(0)
    return {
        'agentview': rng.randint(0, 255, (n_frames, size, size, 3), dtype=np.uint8),
        'wrist': rng.randint(0, 255, (n_frames, size, size, 3), dtype=np.uint8),
        'proprio': rng.rand(n_frames, 9).astype(np.float32),
    }


# ------------------------------------------------------------------ DDIM loop
@requires_ros2
def test_ddim_runs_every_timestep_in_order():
    from evh_controller.dp_onnx_policy import ddim_sample

    seen = []

    def denoise(sample, t, cond):
        seen.append(t)
        return np.zeros_like(sample)

    ddim_sample(np.zeros((1, 16, 10), np.float32), np.zeros((1, 274), np.float32),
                META, denoise)
    assert seen == META['timesteps'], 'the baked timestep sequence must be followed exactly'


@requires_ros2
def test_ddim_zero_noise_prediction_rescales_by_the_alpha_ratio():
    """With eps = 0 the update collapses to x <- sqrt(a_prev/a_t) * x, which is checkable by hand
    and catches an alpha index off by one — the kind of error that still produces plausible
    trajectories."""
    from evh_controller.dp_onnx_policy import ddim_sample

    alphas = np.asarray(META['alphas_cumprod'])
    x0 = np.full((1, 1, 1), 0.5, np.float32)
    meta = {**META, 'timesteps': [75], 'clip_sample': False}

    out = ddim_sample(x0, None, meta, lambda s, t, c: np.zeros_like(s))

    expected = 0.5 * np.sqrt(alphas[75 - 25] / alphas[75])
    assert out.shape == x0.shape
    assert np.allclose(out[0, 0, 0], expected, atol=1e-5)


@requires_ros2
def test_ddim_uses_the_final_alpha_when_the_previous_timestep_falls_off_the_schedule():
    """t = 0 steps to prev_t = -25, which is not an index into alphas_cumprod. Wrapping around
    (numpy's default for a negative index) would silently use alphas_cumprod[75]."""
    from evh_controller.dp_onnx_policy import ddim_sample

    alphas = np.asarray(META['alphas_cumprod'])
    x0 = np.full((1, 1, 1), 0.5, np.float32)
    meta = {**META, 'timesteps': [0], 'clip_sample': False}

    out = ddim_sample(x0, None, meta, lambda s, t, c: np.zeros_like(s))

    assert np.allclose(out[0, 0, 0], 0.5 * np.sqrt(1.0 / alphas[0]), atol=1e-5)


@requires_ros2
def test_ddim_rejects_a_scheduler_it_does_not_implement():
    """A v-prediction checkpoint would otherwise be sampled with the epsilon update and produce
    confident nonsense."""
    from evh_controller.dp_onnx_policy import ddim_sample

    with pytest.raises(NotImplementedError, match='epsilon'):
        ddim_sample(np.zeros((1, 2, 2), np.float32), None,
                    {**META, 'prediction_type': 'v_prediction'},
                    lambda s, t, c: np.zeros_like(s))


@requires_ros2
def test_ddim_guidance_is_reimposed_after_the_last_step():
    """RTC's frozen prefix must survive the loop exactly, not approximately — the executor treats
    those entries as already committed."""
    from evh_controller.dp_onnx_policy import ddim_sample

    guide = np.full((1, 16, 10), 7.0, np.float32)
    mask = np.zeros((1, 16, 1), np.float32)
    mask[0, :3, 0] = 1.0

    out = ddim_sample(np.zeros((1, 16, 10), np.float32), None, META,
                      lambda s, t, c: np.zeros_like(s), guide=guide, mask=mask)

    assert np.allclose(out[0, :3], 7.0), 'a fully-weighted prefix entry was not held'
    assert not np.allclose(out[0, 3:], 7.0), 'guidance leaked past the mask'


# ------------------------------------------------------------- obs plumbing
@requires_ros2
def test_feeds_have_the_shapes_the_encoder_graph_declares():
    b = _stub_backend()
    feeds = b._feeds(_obs())

    assert feeds['agentview'].shape == (2, 3, 84, 84)
    assert feeds['wrist'].shape == (2, 3, 84, 84)
    assert feeds['eef_pos'].shape == (2, 3)
    assert feeds['eef_quat'].shape == (2, 4)
    assert feeds['gripper'].shape == (2, 2)
    for name, arr in feeds.items():
        assert arr.dtype == np.float32, f'{name} is {arr.dtype}, the graph wants float32'


@requires_ros2
def test_images_are_scaled_to_unit_range_not_left_as_bytes():
    """The graph applies the checkpoint's normalizer (x*2 - 1), which assumes [0, 1] input. Feeding
    raw 0..255 would land the observation ~250x outside the encoder's training distribution."""
    b = _stub_backend()
    feeds = b._feeds(_obs())

    assert 0.0 <= feeds['agentview'].min() and feeds['agentview'].max() <= 1.0


@requires_ros2
def test_short_history_is_left_padded_by_repeating_the_oldest_frame():
    """Matches dp_repo_policy._pad_history. Padding with zeros instead would show the policy a
    black frame at every episode start."""
    b = _stub_backend()
    obs = _obs(n_frames=1)

    feeds = b._feeds(obs)

    assert feeds['agentview'].shape[0] == 2
    assert np.array_equal(feeds['agentview'][0], feeds['agentview'][1])
    assert np.array_equal(feeds['eef_pos'][0], feeds['eef_pos'][1])


@requires_ros2
def test_only_the_newest_frames_are_kept_when_history_is_long():
    b = _stub_backend()
    obs = _obs(n_frames=5)

    feeds = b._feeds(obs)

    assert feeds['agentview'].shape[0] == 2
    assert np.allclose(feeds['eef_pos'][-1], obs['proprio'][-1, 0:3])


@requires_ros2
def test_proprio_is_split_into_the_same_slices_the_checkpoint_was_trained_on():
    """[eef_pos(3), eef_quat(4), gripper_qpos(2)] — the observation contract. An off-by-one here
    hands the encoder a quaternion built from a position component."""
    b = _stub_backend()
    obs = _obs()

    feeds = b._feeds(obs)

    assert np.allclose(feeds['eef_pos'], obs['proprio'][:, 0:3])
    assert np.allclose(feeds['eef_quat'], obs['proprio'][:, 3:7])
    assert np.allclose(feeds['gripper'], obs['proprio'][:, 7:9])


@requires_ros2
def test_a_mismatched_camera_resolution_is_resized_to_the_checkpoint_shape():
    b = _stub_backend()

    feeds = b._feeds(_obs(size=96))

    assert feeds['agentview'].shape == (2, 3, 84, 84)


# ----------------------------------------------------------------- unnormalize
@requires_ros2
def test_chunk_starts_at_the_current_observation_step_and_is_seven_dim():
    """The trajectory covers the whole horizon including the past To-1 steps; executing from 0
    would replay actions for observations already consumed."""
    from evh_controller.rotation import abs10_to_abs7

    b = _stub_backend()
    traj = np.arange(16 * 10, dtype=np.float32).reshape(1, 16, 10)

    chunk = b._unnormalize(traj)

    assert chunk.shape == (15, 7)
    assert np.allclose(chunk, abs10_to_abs7(traj[0, 1:]))


@requires_ros2
def test_delta_mode_leaves_the_raw_action_width_alone():
    b = _stub_backend(absolute_actions=False, _raw_action_dim=7, action_dim=7,
                      _action_scale=np.ones(7, np.float32),
                      _action_offset=np.zeros(7, np.float32))
    traj = np.arange(16 * 7, dtype=np.float32).reshape(1, 16, 7)

    chunk = b._unnormalize(traj)

    assert np.allclose(chunk, traj[0, 1:])


@requires_ros2
def test_unnormalize_inverts_the_normalizer_not_reapplies_it():
    """LinearNormalizer.normalize is x*scale + offset, so the inverse is (x - offset)/scale.
    Getting this backwards keeps the shapes right and scales every command wrong."""
    b = _stub_backend(_action_scale=np.full(10, 4.0, np.float32),
                      _action_offset=np.full(10, 1.0, np.float32),
                      absolute_actions=False, _raw_action_dim=10, action_dim=10)
    traj = np.full((1, 16, 10), 9.0, np.float32)

    chunk = b._unnormalize(traj)

    assert np.allclose(chunk, 2.0)   # (9 - 1) / 4
