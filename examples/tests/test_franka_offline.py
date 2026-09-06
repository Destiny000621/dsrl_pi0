"""Franka DSRL learner gates — no robot, no serve, no operator.

Run before every session; it is cheap and it catches the failures that would
otherwise surface only after several episodes of real robot time.

    ~/venvs/dsrl/bin/python -m pytest examples/tests/test_franka_offline.py -q

Runs on CPU (forced below), so it is safe to run with a serve and a live session
already on the GPU.

The load-bearing one is the `(50, 32)` action space. jaxrl2's actor emits a FLAT
1600-D action while the replay buffer stores `(50, 32)`; they only meet because
`networks/mlp.py:_flatten_dict` reshapes any >2-D `actions` entry. `noise_rows=50`
is not optional on this checkpoint (chunk-vs-expert position MAE is 40.7 mm at one
tiled row against 3.3 mm at 50), so that path has to hold.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Force CPU. These are shape/plumbing gates, and the whole point is to run them
# BEFORE a session -- when the pi0.5 serve (~24.6 GB) and possibly a live learner
# already own the GPU. On CUDA this suite OOMs against a running session, which
# would make the gate unusable exactly when you need it. The nets here are tiny.
os.environ.setdefault("JAX_PLATFORMS", "cpu")


@pytest.fixture(scope="module")
def learner():
    from examples.launch_train_franka import build_variant
    from examples.train_franka_service import Learner

    argv = sys.argv
    sys.argv = [
        "test", "--warmup_grad_steps", "12", "--multi_grad_step", "2",
        "--num_initial_traj_collect", "2", "--batch_size", "8", "--max_steps", "2000",
        "--wandb_project", "", "--outputdir", "/tmp/dsrl_test",
    ]
    try:
        v = build_variant(argparse.ArgumentParser())
    finally:
        sys.argv = argv
    return Learner(v), v


def _obs(v, rng):
    c = 3 * v.train_kwargs["num_cameras"]
    return (rng.integers(0, 255, (v.resize_image, v.resize_image, c), dtype=np.uint8),
            rng.standard_normal(v.state_dim).astype(np.float32))


def test_defaults_are_the_measured_ones(learner):
    _, v = learner
    assert v.noise_rows == 50, "one tiled row is 6.7x off this checkpoint's SFT manifold"
    assert v.state_dim == 2058, "10 proprio + 2048 z_rl, measured on the serve"
    assert v.query_freq == 50, "one decision per FULL chunk, as in both upstream recipes"
    assert v.train_kwargs["num_cameras"] == 2, "Franka has wrist + side, not 3 cameras"


def test_warmup_emits_true_gaussian_then_the_actor_takes_over(learner):
    """Episodes before the first update run N(0,1) — upstream's `i == 0` branch.

    Those episodes ARE the frozen-pi0.5 baseline the whole run is judged against,
    so they must not come from the (bounded, tanh-squashed) actor.
    """
    L, v = learner
    rng = np.random.default_rng(0)
    M = v.train_kwargs["action_magnitude"]

    warm = []
    for ep in (1, 2):
        for _ in range(5):
            noise, base = L.infer(ep, *_obs(v, rng))
            assert noise.shape == (v.noise_rows, 32)
            assert base, "warmup episodes must be base-policy"
            warm.append(noise)
        px, st = _obs(v, rng)
        L.close_episode(ep, ep == 2, px, st, 250)

    # N(0,1) is unbounded, so the warmup noise should exceed the actor's bound.
    assert np.abs(np.stack(warm)).max() > M, "warmup noise looks bounded — not N(0,1)"

    noise, base = L.infer(3, *_obs(v, rng))
    assert not base, "actor should be live once updates have run"
    assert np.abs(noise).max() <= M + 1e-4, f"actor exceeded action_magnitude {M}"


def test_gradient_steps_run_on_the_50x32_action(learner):
    """The (50,32) buffer action must survive jaxrl2's flatten path to 1600-D."""
    L, v = learner
    rng = np.random.default_rng(1)
    for _ in range(5):
        L.infer(4, *_obs(v, rng))
    px, st = _obs(v, rng)
    out = L.close_episode(4, False, px, st, 250)
    assert out["grad_steps"] == 5 * v.multi_grad_step
    assert out["buffer_size"] > 0


def test_aborted_episodes_never_reach_the_buffer(learner):
    """DSRL cannot learn from interventions: there is no action->noise inverse for
    a frozen flow policy, so a taken-over episode must be dropped, not relabelled."""
    L, v = learner
    rng = np.random.default_rng(2)
    before = L.health()["buffer_size"]
    for _ in range(3):
        L.infer(99, *_obs(v, rng))
    assert L.abort_episode(99)["dropped"] == 3
    assert L.health()["buffer_size"] == before


def test_wrong_state_width_fails_loudly(learner):
    """A silently wrong z_rl width would poison every transition in the buffer."""
    L, v = learner
    rng = np.random.default_rng(3)
    pixels, _ = _obs(v, rng)
    with pytest.raises(ValueError, match="state is"):
        L.infer(5, pixels, np.zeros(v.state_dim - 1, np.float32))


def test_save_and_restore_round_trip(tmp_path):
    """Ctrl+C must not throw away robot time, and the resume must be a real one.

    Upstream never persists the buffer on the real-robot path. At 100 episodes
    those transitions ARE the run, so this covers the whole cycle: collect ->
    save_all -> fresh Learner -> restore -> same size, same counters, still
    samplable (a restored buffer that cannot produce a batch is not a resume).
    """
    from examples.launch_train_franka import build_variant
    from examples.train_franka_service import Learner

    argv = sys.argv
    sys.argv = [
        "test", "--warmup_grad_steps", "4", "--multi_grad_step", "1",
        "--num_initial_traj_collect", "1", "--batch_size", "4", "--max_steps", "2000",
        "--wandb_project", "", "--outputdir", str(tmp_path),
    ]
    try:
        v = build_variant(argparse.ArgumentParser())
    finally:
        sys.argv = argv

    a = Learner(v)
    rng = np.random.default_rng(7)
    for ep in (1, 2):
        for _ in range(4):
            a.infer(ep, *_obs(v, rng))
        px, st = _obs(v, rng)
        a.close_episode(ep, ep == 2, px, st, 200)
    saved = a.save_all("test")
    assert "buffer" in saved, saved
    buf = tmp_path / "replay_buffer.pkl"
    assert buf.exists() and not (tmp_path / "replay_buffer.pkl.tmp").exists()

    before = a.health()

    # Restore WEIGHTS + buffer together — the counter travels with the weights
    # (restoring the buffer alone now deliberately resets grad_steps to re-warm;
    # that path has its own test below).
    sys.argv = [
        "test", "--warmup_grad_steps", "4", "--multi_grad_step", "1",
        "--num_initial_traj_collect", "1", "--batch_size", "4", "--max_steps", "2000",
        "--wandb_project", "", "--outputdir", str(tmp_path),
        "--restore_buffer", str(buf), "--restore_path", str(tmp_path),
    ]
    try:
        v2 = build_variant(argparse.ArgumentParser())
    finally:
        sys.argv = argv
    b = Learner(v2)
    after = b.health()

    assert after["buffer_size"] == before["buffer_size"] > 0
    assert after["total_traj"] == before["total_traj"]
    assert after["grad_steps"] == before["grad_steps"]
    assert after["success_rate_10"] == before["success_rate_10"]
    # A restored buffer must still yield a usable batch.
    batch = next(b.buffer.get_iterator(2))
    assert batch["actions"].shape == (2, v.noise_rows, 32)
    assert batch["observations"]["state"].shape[0] == 2


def test_stale_staged_episodes_are_dropped(learner):
    """A robot session that dies mid-episode leaves staged decisions behind, and
    the NEXT session restarts its episode numbering. Without the guard the stale
    list could collide with a reused id and splice two episodes into one
    trajectory. First decision under a new id must flush anything stale."""
    L, v = learner
    rng = np.random.default_rng(11)
    for _ in range(3):
        L.infer(300, *_obs(v, rng))       # episode 300 never closes (crash)
    assert 300 in L.staged
    L.infer(301, *_obs(v, rng))           # new session's first decision
    assert 300 not in L.staged, "stale episode must be flushed"
    assert len(L.staged[301]) == 1
    L.abort_episode(301)


def _variant(tmp_path, *extra):
    from examples.launch_train_franka import build_variant

    argv = sys.argv
    sys.argv = [
        "test", "--warmup_grad_steps", "4", "--multi_grad_step", "1",
        "--num_initial_traj_collect", "1", "--batch_size", "4", "--max_steps", "2000",
        "--wandb_project", "", "--outputdir", str(tmp_path), *extra,
    ]
    try:
        return build_variant(argparse.ArgumentParser())
    finally:
        sys.argv = argv


def test_outputdir_is_absolute(tmp_path):
    """flax refuses relative checkpoint paths — with a relative EXP the buffer
    saved but the WEIGHTS silently did not (live 2026-09-05, zero checkpoints
    after 6680 grad steps)."""
    import examples.launch_train_franka as l

    argv = sys.argv
    sys.argv = ["test", "--wandb_project", "", "--outputdir", "logs/rel/x"]
    try:
        v = l.build_variant(argparse.ArgumentParser())
    finally:
        sys.argv = argv
    assert os.path.isabs(v.outputdir)


def test_eval_requires_restore_path(tmp_path):
    with pytest.raises(SystemExit, match="restore_path"):
        _variant(tmp_path, "--eval", "1")


def test_eval_mode_never_learns_and_never_warms_up(tmp_path):
    """Eval: the actor answers every decision (no N(0,1) warmup) and nothing is
    inserted, updated, or saved — the running success count IS the output."""
    from examples.train_franka_service import Learner

    # train 2 tiny episodes so a real checkpoint exists to restore
    v = _variant(tmp_path)
    t = Learner(v)
    rng = np.random.default_rng(5)
    for ep in (1, 2):
        for _ in range(3):
            t.infer(ep, *_obs(v, rng))
        px, st = _obs(v, rng)
        t.close_episode(ep, ep == 2, px, st, 150)
    t.agent.save_checkpoint(v.outputdir, t.grad_steps, 1)

    ve = _variant(tmp_path, "--eval", "1", "--restore_path", str(tmp_path))
    e = Learner(ve)
    M = ve.train_kwargs["action_magnitude"]
    for _ in range(3):
        noise, base = e.infer(7, *_obs(ve, rng))
        assert not base, "eval must never fall back to N(0,1)"
        assert np.abs(noise).max() <= M + 1e-4, "eval noise must come from the actor"
    px, st = _obs(ve, rng)
    out = e.close_episode(7, True, px, st, 150)
    assert out.get("eval") and out["decisions"] == 3 and out["success_rate"] == 1.0
    assert e.health()["mode"] == "eval"
    assert e.grad_steps == 0 and len(e.buffer) == 0, "eval inserted or learned"
    assert e.save_all("test") == {}, "eval must never touch the training run's files"


def test_buffer_restore_without_weights_rewarms(tmp_path):
    """grad_steps describes the WEIGHTS. Restoring the counter without
    --restore_path would report base_policy=False while a fresh random actor
    drives the arm; instead the counter resets so the warmup block retrains
    from the restored buffer (the 2026-09-05 recovery path)."""
    from examples.train_franka_service import Learner

    v = _variant(tmp_path)
    t = Learner(v)
    rng = np.random.default_rng(6)
    for ep in (1, 2):
        for _ in range(3):
            t.infer(ep, *_obs(v, rng))
        px, st = _obs(v, rng)
        t.close_episode(ep, False, px, st, 150)
    assert t.grad_steps > 0
    t.save_all("test")

    v2 = _variant(tmp_path, "--restore_buffer", str(tmp_path / "replay_buffer.pkl"))
    r = Learner(v2)
    assert len(r.buffer) == len(t.buffer)
    assert r.total_traj == 2
    assert r.grad_steps == 0, "no weights restored -> counter must reset (re-warm)"
    assert r.health()["base_policy"] is True
