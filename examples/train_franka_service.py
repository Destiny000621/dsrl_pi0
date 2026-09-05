#!/usr/bin/env python3
"""DSRL learner service for the Franka FR3 (avantbot station).

Upstream's real-robot entry point (`examples/train_real.py` +
`train_utils_real.py`) owns BOTH the SAC learner and the robot loop. Here the
robot loop lives in avantbot (agent `franka_pi05_dsrl`, driven by
`python -m avantbot.collect`), so this process keeps only the learner half:
the jaxrl2 PixelSAC agent, the replay buffer, and upstream's episodic
insert/update schedule. **`jaxrl2/` is not modified** — algorithm frozen by
scope; every difference from upstream is embodiment plumbing.

Mapping onto `trajwise_alternating_training_loop`:

    upstream                                   here
    ------------------------------------------ -------------------------------
    collect_traj() inner loop, t%query_freq==0  POST /infer  (one per decision)
    keypress 1/0 + reward/mask construction     POST /episode (operator label)
    add_online_data_to_buffer + grad-step block POST /episode (same call)
    `if i == 0` -> N(0,1) noise                 /infer answers N(0,1) until the
                                                first update has run

The observations that enter the buffer are the SAME arrays the actor was asked
about: /infer stages (obs, noise) per episode, and /episode only supplies the
labels plus the terminal observation. Nothing is re-sent, so there is no way for
the buffer's obs and the actor's obs to drift apart.

Wire format: msgpack_numpy over HTTP (the convention the SubRL learner on this
station already uses, so the robot side shares one serialization helper).

    POST /infer    {episode_id, step_id, pixels (R,R,3*C) u8, state (S,) f32}
                -> {noise (H,32) f32, base_policy bool, param_version int}
    POST /episode  {episode_id, is_success bool, pixels, state, env_steps}
                -> {decisions, grad_steps, buffer_size, total_traj}
    POST /abort    {episode_id}                  -> drop staged, nothing learned
    GET  /healthz                                -> counters
"""

from __future__ import annotations

import argparse
import http.server
import json
import logging
import os
import pickle
import signal
import socketserver
import tempfile
import threading
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("dsrl")


class Learner:
    """jaxrl2 PixelSAC + replay buffer + upstream's episodic update schedule."""

    def __init__(self, variant):
        import jax  # noqa: PLC0415
        from gym.spaces import Box, Dict  # noqa: PLC0415
        from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner  # noqa: PLC0415
        from jaxrl2.data import ReplayBuffer  # noqa: PLC0415
        from jaxrl2.utils.general_utils import add_batch_dim  # noqa: PLC0415

        self.v = variant
        self.lock = threading.Lock()

        # --- spaces -------------------------------------------------------
        # pixels: every camera channel-concatenated, then a trailing frame-stack
        # axis of 1 (jaxrl2 convention). Franka has TWO cameras -> 6 channels,
        # where both upstream recipes assume 3 cameras / 9 channels.
        R = variant.resize_image
        self.image_shape = (R, R, 3 * variant.num_cameras, 1)
        obs_dict = {"pixels": Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)}
        # state = [proprio(10), z_rl(emb)]. NEVER a hardcoded constant: upstream's
        # `state_dim = 8 + 2024` is wrong twice over. Measured on this serve: 2058.
        obs_dict["state"] = Box(low=-np.inf, high=np.inf, shape=(variant.state_dim, 1), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        # THE action is the flow-matching latent. noise_rows=50 (the full chunk) is
        # forced by measurement -- one tiled row is 40.7mm chunk-vs-expert position
        # MAE against 3.3mm at 50 (openpi scripts/dsrl_verify_noise.py --rows-sweep).
        # jaxrl2 reads action_chunk_shape off this Box and flattens to 50*32=1600
        # inside the nets (networks/mlp.py:_flatten_dict), so this is the only place
        # the choice appears.
        self.action_space = Box(
            low=-variant.action_magnitude,
            high=variant.action_magnitude,
            shape=(variant.noise_rows, 32),
            dtype=np.float32,
        )

        sample_obs = add_batch_dim(self.observation_space.sample())
        sample_action = add_batch_dim(self.action_space.sample())
        logger.info("obs spaces: %s", [(k, v.shape) for k, v in sample_obs.items()])
        logger.info("action space: %s -> flat %d", sample_action.shape, variant.noise_rows * 32)

        kwargs = dict(variant.train_kwargs)
        if kwargs.pop("cosine_decay", False):
            kwargs["decay_steps"] = variant.max_steps
        self.agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **kwargs)
        if variant.restore_path:  # upstream references variant.restore_path but never
            logger.info("restoring actor/critic from %s", variant.restore_path)  # adds the flag (B10a)
            self.agent.restore_checkpoint(variant.restore_path)

        capacity = int(2 * variant.max_steps // variant.multi_grad_step)
        self.buffer = ReplayBuffer(self.observation_space, self.action_space, capacity)
        self.buffer.seed(variant.seed)
        self.iterator = self.buffer.get_iterator(variant.batch_size)
        logger.info(
            "replay capacity %d transitions (~%.1f GB); expected fill from %d episodes: ~%d",
            capacity,
            capacity * (2 * np.prod(self.image_shape) + 2 * variant.state_dim * 4 + 2 * variant.noise_rows * 32 * 4) / 1e9,
            variant.num_episodes,
            variant.num_episodes * 40,
        )

        self.jax = jax
        self.rng = jax.random.PRNGKey(variant.seed + 1)
        self.grad_steps = 0          # upstream's `i`
        self.total_traj = 0
        self.total_env_steps = 0
        self.param_version = 0
        self.staged: dict[int, list] = {}   # episode_id -> [(obs_dict, noise), ...]
        self.successes: list[int] = []
        self.wandb = _make_wandb(variant)
        if variant.restore_buffer:
            self._restore_buffer(variant.restore_buffer)

    # -- decisions ---------------------------------------------------------
    def infer(self, episode_id: int, pixels: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, bool]:
        """One DSRL decision: stage the observation, return the latent to execute."""
        if state.shape[-1] != self.v.state_dim:
            raise ValueError(
                f"state is {state.shape[-1]}-D but the learner was built for "
                f"{self.v.state_dim}. Re-measure z_rl (openpi dsrl_verify_noise.py G3) "
                "and relaunch with --state-dim; a silently wrong width poisons every "
                "transition in the buffer."
            )
        obs = {
            "pixels": np.asarray(pixels, np.uint8).reshape(self.image_shape)[None],
            "state": np.asarray(state, np.float32).reshape(-1, 1)[None],
        }
        with self.lock:
            base_policy = self.grad_steps == 0
            if base_policy:
                # Upstream's `if i == 0` branch: pure N(0,1) until the first update.
                # These episodes ARE the frozen-pi0.5 baseline the run is judged against.
                self.rng, key = self.jax.random.split(self.rng)
                noise = np.asarray(
                    self.jax.random.normal(key, (self.v.noise_rows, 32)), np.float32
                )
            else:
                noise = np.reshape(
                    self.agent.sample_actions(obs), (self.v.noise_rows, 32)
                ).astype(np.float32)
            self.staged.setdefault(episode_id, []).append((obs, noise))
        return noise, base_policy

    # -- episode close -----------------------------------------------------
    def close_episode(self, episode_id: int, is_success: bool, pixels, state, env_steps: int) -> dict:
        with self.lock:
            steps = self.staged.pop(episode_id, [])
            if not steps:
                return {"decisions": 0, "grad_steps": 0, "buffer_size": len(self.buffer), "note": "empty"}

            terminal = {
                "pixels": np.asarray(pixels, np.uint8).reshape(self.image_shape)[None],
                "state": np.asarray(state, np.float32).reshape(-1, 1)[None],
            }
            n = len(steps)
            # Upstream's sparse per-decision convention (train_utils_real.py:250-257):
            # success -> [-1..-1, 0] with the LAST mask 0 (absorbing goal, no bootstrap);
            # failure/timeout -> all -1, all masks 1 (bootstraps THROUGH the time limit,
            # so a timeout is not mistaken for a terminal state).
            if is_success:
                rewards = np.concatenate([-np.ones(n - 1), [0.0]])
                masks = np.concatenate([np.ones(n - 1), [0.0]])
            else:
                rewards = -np.ones(n)
                masks = np.ones(n)

            obs_list = [o for o, _ in steps] + [terminal]
            actions = np.stack([a for _, a in steps])
            discount = self.v.discount ** self.v.query_freq
            for t in range(n):
                self.buffer.insert(
                    dict(
                        observations={k: v[0] for k, v in obs_list[t].items()},
                        next_observations={k: v[0] for k, v in obs_list[t + 1].items()},
                        actions=actions[t],
                        next_actions=actions[t + 1] if t < n - 1 else actions[t],
                        rewards=rewards[t],
                        masks=masks[t],
                        discount=discount,
                    )
                )
            self.buffer.increment_traj_counter()
            self.total_traj += 1
            self.total_env_steps += int(env_steps)
            self.successes.append(int(bool(is_success)))

            done = 0
            if self.total_traj >= self.v.num_initial_traj_collect and self.grad_steps <= self.v.max_steps:
                # Upstream: a big first block, then UTD * decisions per episode.
                n_grad = self.v.warmup_grad_steps if self.grad_steps == 0 else n * self.v.multi_grad_step
                t0 = time.time()
                for _ in range(n_grad):
                    info = self.agent.update(next(self.iterator))
                    self.grad_steps += 1
                    done += 1
                    if self.grad_steps % self.v.log_interval == 0 and self.wandb is not None:
                        self._log_update(info)
                    if (
                        self.v.checkpoint_interval > 0
                        and self.grad_steps % self.v.checkpoint_interval == 0
                    ):
                        self.agent.save_checkpoint(
                            self.v.outputdir, self.grad_steps, self.v.checkpoint_interval
                        )
                self.param_version += 1
                logger.info("episode %d: %d grad steps in %.1fs", self.total_traj, done, time.time() - t0)

            self._log_episode(is_success)
            if self.v.save_every_episodes > 0 and self.total_traj % self.v.save_every_episodes == 0:
                self.save_all(f"episode {self.total_traj}")
            return {
                "decisions": n,
                "grad_steps": done,
                "total_grad_steps": self.grad_steps,
                "buffer_size": len(self.buffer),
                "total_traj": self.total_traj,
                "success_rate_10": float(np.mean(self.successes[-10:])),
            }

    def abort_episode(self, episode_id: int) -> dict:
        """Drop an unrecoverable episode: its steps never enter the buffer.

        DSRL cannot consume human interventions (there is no action->noise inverse
        for a frozen flow policy), so a takeover episode must be discarded, not
        relabelled -- inserting it would silently corrupt the baseline's
        no-intervention semantics.
        """
        with self.lock:
            dropped = len(self.staged.pop(episode_id, []))
        logger.info("episode %s ABORTED, %d staged decisions dropped", episode_id, dropped)
        return {"dropped": dropped}

    def health(self) -> dict:
        return {
            "total_traj": self.total_traj,
            "grad_steps": self.grad_steps,
            "buffer_size": len(self.buffer),
            "param_version": self.param_version,
            "base_policy": self.grad_steps == 0,
            "success_rate_10": float(np.mean(self.successes[-10:])) if self.successes else 0.0,
            "episodes_target": self.v.num_episodes,
        }

    # -- persistence -------------------------------------------------------
    def save_all(self, reason: str) -> dict:
        """Checkpoint actor/critic AND the replay buffer.

        Upstream never persists the buffer on the real-robot path, which is
        survivable at its 500k-gradstep budget and NOT survivable at 100 real
        episodes: those transitions cost robot time and cannot be regenerated.
        Called periodically, and on SIGINT/SIGTERM so Ctrl+C is not destructive.
        """
        out = {}
        try:
            self.agent.save_checkpoint(self.v.outputdir, self.grad_steps, self.v.checkpoint_interval)
            out["checkpoint"] = f"{self.v.outputdir}/checkpoint_{self.grad_steps}"
        except Exception as exc:  # noqa: BLE001 — a duplicate step must not lose the buffer
            logger.warning("actor/critic checkpoint at step %d not written (%s)", self.grad_steps, exc)
        try:
            path = os.path.join(self.v.outputdir, "replay_buffer.pkl")
            tmp = path + ".tmp"
            # Write-then-rename: a Ctrl+C during the write must not leave a
            # half-written buffer where the previous good one used to be.
            self.buffer.save(tmp)
            with open(os.path.join(self.v.outputdir, "counters.json"), "w") as f:
                json.dump(
                    {"total_traj": self.total_traj, "grad_steps": self.grad_steps,
                     "total_env_steps": self.total_env_steps, "successes": self.successes},
                    f,
                )
            os.replace(tmp, path)
            out["buffer"] = f"{path} ({len(self.buffer)} transitions)"
        except Exception as exc:  # noqa: BLE001
            logger.error("replay buffer NOT saved (%s)", exc)
        logger.info("saved [%s]: %s", reason, out or "nothing")
        return out

    def _restore_buffer(self, path: str) -> None:
        """Reload a saved buffer + counters.

        Deliberately plain `pickle.load` rather than jaxrl2's
        `ReplayBuffer.restore`, which does `np.load(..., allow_pickle=True)[0]`
        on a `pickle.dump` file and is marked "todo test this" upstream.
        """
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.buffer.data = d["data"]
        self.buffer.size = d["size"]
        self.buffer._traj_counter = d["_traj_counter"]  # noqa: SLF001
        self.buffer._start = d["_start"]  # noqa: SLF001
        self.buffer.traj_bounds = d["traj_bounds"]
        counters = os.path.join(os.path.dirname(path), "counters.json")
        if os.path.exists(counters):
            with open(counters) as f:
                c = json.load(f)
            self.total_traj = c["total_traj"]
            self.grad_steps = c["grad_steps"]
            self.total_env_steps = c["total_env_steps"]
            self.successes = c["successes"]
        logger.info("restored buffer: %d transitions, %d episodes, %d grad steps",
                    len(self.buffer), self.total_traj, self.grad_steps)

    # -- logging -----------------------------------------------------------
    def _log_update(self, info) -> None:
        info = {k: self.jax.device_get(v) for k, v in info.items()}
        for k, val in info.items():
            if getattr(val, "ndim", 0) == 0:
                self.wandb.log({f"training/{k}": val}, step=self.grad_steps)

    def _log_episode(self, is_success: bool) -> None:
        if self.wandb is None:
            return
        self.wandb.log(
            {
                "is_success": int(bool(is_success)),
                "total_num_traj": self.total_traj,
                "env_steps": self.total_env_steps,
                "replay_buffer_size": len(self.buffer),
                # Parity aliases with the SubRL learner so the two baselines can be
                # read off one wandb dashboard.
                "episode_reward": float(bool(is_success)),
                "success_rate_10": float(np.mean(self.successes[-10:])),
                "success_rate_20": float(np.mean(self.successes[-20:])),
            },
            step=self.grad_steps,
        )


def _make_wandb(variant):
    if not variant.wandb_project:
        logger.warning("wandb disabled (--wandb-project ''); the run will not be comparable")
        return None
    from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name  # noqa: PLC0415

    expname = create_exp_name(variant.prefix, seed=variant.seed)
    return WandBLogger(
        True, variant, variant.wandb_project, experiment_id=expname,
        output_dir=tempfile.mkdtemp(), group_name=variant.wandb_group,
    )


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class _Handler(http.server.BaseHTTPRequestHandler):
    learner: Learner = None  # set on the class before serve_forever

    def log_message(self, *args):  # noqa: D102 — silence per-request stderr spam
        pass

    def _send(self, obj, code=200):
        from openpi_client import msgpack_numpy  # noqa: PLC0415

        body = msgpack_numpy.Packer().pack(obj)
        self.send_response(code)
        self.send_header("Content-Type", "application/msgpack")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/healthz"):
            body = json.dumps(self.learner.health()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):  # noqa: N802
        from openpi_client import msgpack_numpy  # noqa: PLC0415

        try:
            payload = msgpack_numpy.unpackb(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.startswith("/infer"):
                noise, base = self.learner.infer(
                    int(payload["episode_id"]), payload["pixels"], payload["state"]
                )
                self._send({"noise": noise, "base_policy": base,
                            "param_version": self.learner.param_version})
            elif self.path.startswith("/episode"):
                self._send(self.learner.close_episode(
                    int(payload["episode_id"]), bool(payload["is_success"]),
                    payload["pixels"], payload["state"], int(payload.get("env_steps", 0)),
                ))
            elif self.path.startswith("/abort"):
                self._send(self.learner.abort_episode(int(payload["episode_id"])))
            else:
                self.send_error(404)
        except Exception as exc:  # noqa: BLE001
            logger.exception("request failed")
            self._send({"error": repr(exc)}, code=500)


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main(variant) -> None:
    os.makedirs(variant.outputdir, exist_ok=True)
    _Handler.learner = Learner(variant)
    logger.info("DSRL learner ready on :%d — waiting for the robot loop", variant.port)
    logger.info("base-policy phase: first %d episode(s) run N(0,1) noise (frozen pi0.5)",
                variant.num_initial_traj_collect)
    server = _Server(("127.0.0.1", variant.port), _Handler)

    def _shutdown(signum, _frame):  # noqa: ANN001
        # Ctrl+C during a 100-episode session must not throw away robot time.
        logger.info("signal %d — saving before exit", signum)
        _Handler.learner.save_all(f"signal {signum}")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(
        "Launch via `python -m examples.launch_train_franka` — that module "
        "owns the argparse surface and the hyperparameter rationale."
    )
