"""End-to-end smoke of the DSRL training loop WITHOUT the robot (docs/yam_verification.md).

REAL: the launcher's argparse, train_yam.main, PixelSACLearner construction + JIT on the
GPU (beside the serve), the live pi0.5 serve (get_prefix_rep + noise infer), collect_traj,
replay insert/sample, the 5000-step warmup and per-episode update blocks, checkpoint save,
the video writer, --max_episodes stop, and a second pass that restores the checkpoint.
FAKE: the robot (synthetic obs of the exact limb shapes; records every commanded action),
the keypress gates (episodes end by timeout; labels alternate 1/0), and wandb (no-op).

Run inside the dsrl venv with the serve up:
    python examples/scripts/smoke_loop_no_robot.py
"""
import glob
import itertools
import os
import pathlib
import runpy
import sys
import types

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)


class FakeYamEnv:
    instances = []
    control_hz = 30.0
    gripper_open_cmd = 1.0
    _gripper_clip = (0.0, 1.0)

    def __init__(self, **kw):
        self.kw = kw
        self.rng = np.random.default_rng(0)
        self.cmds = []
        self.resets = 0
        self.q = np.array([0.12, 0.3, 0.8, 0.0, 0.4, 0.0, 1.0, 0.0, 0.3, 0.8, 0.0, 0.4, 0.0, 1.0])
        FakeYamEnv.instances.append(self)

    def _img(self):
        return self.rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)

    def _obs(self):
        return {
            "timestamp": 0.0,
            "left": {"joint_pos": self.q[:6].copy(), "joint_vel": np.zeros(6), "gripper_pos": self.q[6:7].copy()},
            "right": {"joint_pos": self.q[7:13].copy(), "joint_vel": np.zeros(6), "gripper_pos": self.q[13:14].copy()},
            "head_camera": {"images": {"rgb": self._img()}, "timestamp": 0.0},
            "left_wrist_camera": {"images": {"rgb": self._img()}, "timestamp": 0.0},
            "right_wrist_camera": {"images": {"rgb": self._img()}, "timestamp": 0.0},
        }

    def get_observation(self):
        return self._obs()

    def step(self, a):
        a = np.asarray(a, np.float64).reshape(-1)
        assert a.shape == (14,), a.shape
        assert np.isfinite(a).all(), "non-finite action reached the env"
        self.cmds.append(a)
        self.q = 0.9 * self.q + 0.1 * a  # crude tracking toward the target

    def hold(self):
        pass

    def reset(self):
        self.resets += 1
        return self._obs()

    def close(self):
        print(f"[FakeYamEnv] closed: {len(self.cmds)} commands, {self.resets} resets")


class NoWandb:
    logged = []

    def __init__(self, *a, **k):
        pass

    def log(self, d, *a, **k):
        NoWandb.logged.append(dict(d))

    def log_histogram(self, *a, **k):
        pass


# ---- patch BEFORE the launcher imports train_yam (it imports YamEnv at module import,
#      and WandBLogger / the loop helpers inside main()) ----------------------------------
import examples.envs.yam_env as _ye  # noqa: E402
_ye.YamEnv = FakeYamEnv
import jaxrl2.utils.wandb_logger as _wl  # noqa: E402
_wl.WandBLogger = NoWandb
import examples.train_utils_yam as T  # noqa: E402
_labels = itertools.cycle("10")
T._wait_for_key = lambda valid, prompt: ("c" if "c" in valid else next(_labels))
T._drain_stdin = lambda: None
T.termios = types.SimpleNamespace(tcgetattr=lambda fd: None, tcsetattr=lambda *a: None, TCSADRAIN=0)
T.tty = types.SimpleNamespace(setcbreak=lambda fd: None)
T.select = types.SimpleNamespace(select=lambda *a, **k: ([], [], []))

EXP = str(REPO / "logs" / "smoke_no_robot")
os.environ["EXP"] = EXP
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2")
COMMON = [
    "--mode", "keypress", "--wandb_project", "dsrl-smoke",
    "--noise_rows", "50", "--query_freq", "25", "--action_horizon", "50",
    "--max_timesteps", "75",            # 3 decisions per episode
    "--num_initial_traj_collect", "1", "--multi_grad_step", "2",
    "--batch_size", "32", "--buffer_size", "200", "--hidden_dims", "256",
    "--log_interval", "500", "--eval_interval", "2500", "--checkpoint_interval", "2500",
    "--max_steps", "100000",
]


def run(args):
    sys.argv = ["launch_train_yam.py"] + COMMON + args
    try:
        runpy.run_path(str(REPO / "examples" / "launch_train_yam.py"), run_name="__main__")
    except SystemExit:
        pass


# ---- pass 1: fresh run, 3 episodes, warmup + updates, checkpoints -----------------------
run(["--prefix", "smoke_pass1", "--max_episodes", "3"])
env1 = FakeYamEnv.instances[-1]
assert len(env1.cmds) == 3 * 75, f"expected 225 commanded steps, got {len(env1.cmds)}"
assert env1.resets == 2 * 3, f"expected reset at start+end of each episode (6), got {env1.resets}"
run_dirs = sorted(glob.glob(f"{EXP}/smoke_pass1_*"))
assert run_dirs, "no output dir"
out = run_dirs[-1]
ckpts = sorted(glob.glob(f"{out}/checkpoint[0-9]*"))
assert ckpts, f"no checkpoint<step> dir written in {out}"
videos = glob.glob(f"{out}/video_high_*.mp4")
assert len(videos) == 3, f"expected 3 episode videos, got {len(videos)}"
ep_rewards = [d["episode_reward"] for d in NoWandb.logged if "episode_reward" in d]
assert ep_rewards == [1.0, 0.0, 1.0], ep_rewards   # labels cycled 1,0,1
assert not any("success_rate_10" in d for d in NoWandb.logged), "success_rate_10 must wait for 10 episodes"
train_keys = {k for d in NoWandb.logged for k in d if k.startswith("training/")}
assert {"training/critic_loss", "training/actor_loss", "training/temperature"} <= train_keys, train_keys
print(f"\nPASS 1 OK: {len(env1.cmds)} steps, checkpoints {[pathlib.Path(c).name for c in ckpts]}, "
      f"{len(videos)} videos, episode_reward {ep_rewards}")

# ---- pass 2: restore the checkpoint, 1 episode, actor acts from episode 1 ----------------
NoWandb.logged.clear()
run(["--prefix", "smoke_pass2", "--max_episodes", "1", "--restore_path", out])
env2 = FakeYamEnv.instances[-1]
assert len(env2.cmds) == 75, len(env2.cmds)
print("PASS 2 OK: restored actor drove 1 episode")
print("\nEND-TO-END SMOKE (no robot) PASSED")
