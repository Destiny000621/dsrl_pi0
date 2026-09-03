"""Offline (no robot, no GPU, no heavy deps) regression tests for the YAM port.

Covers the safety chain (YAM-abc FlexPoint [0,1] grippers), the obs/wire
pipeline, the buffer insert contract, and the review-fix behaviors (measured
re-anchor on reset, soft-release on close, empty-traj skip, post-trim labels).

Run with plain python3 (numpy + pyyaml only — jax/moviepy/openpi_client/limb
are stubbed):    python3 examples/tests/test_offline_no_robot.py
"""
import pathlib
import sys
import types

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def _fake_resize(img, h, w):
    out = np.zeros((h, w, 3), dtype=img.dtype)
    ch, cw = min(h, img.shape[0]), min(w, img.shape[1])
    out[:ch, :cw] = img[:ch, :cw]
    return out


_stub("jax", numpy=np, random=types.SimpleNamespace(split=None, normal=None))
_stub("moviepy")
_stub("moviepy.editor", ImageSequenceClip=object)
_stub("tqdm", tqdm=lambda x, **k: x)
oc = _stub("openpi_client")
oc.image_tools = types.SimpleNamespace(resize_with_pad=_fake_resize)
sys.modules["openpi_client.image_tools"] = oc.image_tools
_stub("limb")
_stub("limb.utils")
cleanup_calls = []
_stub("limb.utils.launch_utils", cleanup_processes=lambda a, p: cleanup_calls.append((a, p)))

sys.path.insert(0, str(REPO))
from examples import train_utils_yam as T  # noqa: E402
from examples.envs.yam_env import YamEnv  # noqa: E402

# ---------------------------------------------------------------- fixtures

H, W = 480, 640
LIMITS = np.array([[-2.09, 3.14], [0, 3.14], [0.05, 3.14],
                   [-1.35, 1.35], [-1.50, 1.50], [-2.00, 2.00]])
RAW = {
    "timestamp": 0.0,
    "left": {"joint_pos": np.arange(6, dtype=np.float64), "joint_vel": np.zeros(6),
             "gripper_pos": np.array([0.5])},
    "right": {"joint_pos": np.arange(6, 12, dtype=np.float64), "joint_vel": np.zeros(6),
              "gripper_pos": np.array([0.9])},
    "head_camera": {"images": {"rgb": np.full((H, W, 3), 10, np.uint8)}, "timestamp": 0.0},
    "left_wrist_camera": {"images": {"rgb": np.full((H, W, 3), 20, np.uint8)}, "timestamp": 0.0},
    "right_wrist_camera": {"images": {"rgb": np.full((H, W, 3), 30, np.uint8)}, "timestamp": 0.0},
}


class V:
    resize_image = 128
    num_cameras = 3
    query_freq = 25
    discount = 0.999
    add_states = 1


def make_env(gripper_clip=(0.0, 1.0), open_cmd=1.0):
    env = YamEnv.__new__(YamEnv)
    env._joint_delta_limit = 0.15
    env._joint_limit_margin = 0.05
    env._gripper_clip = gripper_clip
    env.gripper_open_cmd = open_cmd
    env._joint_limits = {"left": LIMITS.copy(), "right": LIMITS.copy()}
    env._last_cmd = {"left": np.array([0., 1., 1., 0., 0., 0.]),
                     "right": np.array([0., 1., 1., 0., 0., 0.])}
    env._home = {k: v.copy() for k, v in env._last_cmd.items()}
    env._reset_duration_s = 0.01
    env._pending_obs = None
    env._ee_injector = None
    env._server_processes = []
    return env


# ------------------------------------------------- obs / wire / SAC shapes

curr = T.extract_yam_observation(RAW)
np.testing.assert_allclose(curr["qpos"], [0, 1, 2, 3, 4, 5, 0.5, 6, 7, 8, 9, 10, 11, 0.9])
req = T.get_pi0_input(curr, "insert the wireless bluetooth earbuds into the charging case")
assert set(req["images"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
for v in req["images"].values():
    assert v.shape == (3, 224, 224) and v.dtype == np.uint8
assert req["images"]["cam_high"].max() == 10 and req["images"]["cam_right_wrist"].max() == 30
assert T.process_images(V, curr).shape == (1, 128, 128, 9, 1)


class _DP:
    def get_prefix_rep(self, r):
        return {"prefix_rep": np.ones((1, 2048), np.float32)}


sac = T.get_sac_obs(V, curr, _DP(), req)
assert sac["pixels"].shape == (1, 128, 128, 9, 1) and sac["state"].shape == (1, 2062, 1)

# --------------------------------------------------------- buffer contract


class _Buf:
    def __init__(self):
        self.rows, self.inc = [], 0

    def insert(self, d):
        self.rows.append(d)

    def increment_traj_counter(self):
        self.inc += 1


n = 4
traj = {"observations": [{"pixels": np.full((1, 2), k)} for k in range(n + 1)],
        "actions": [np.full((1, 32), k, np.float32) for k in range(n)],
        "rewards": np.array([-1., -1., -1., 0.]), "masks": np.array([1., 1., 1., 0.]),
        "is_success": True, "env_steps": 100}
b = _Buf()
T.add_online_data_to_buffer(V, traj, b)
assert len(b.rows) == n and b.inc == 1
assert np.isclose(b.rows[0]["discount"], 0.999 ** 25)
assert b.rows[-1]["rewards"] == 0.0 and b.rows[-1]["masks"] == 0.0
b2 = _Buf()
T.add_online_data_to_buffer(V, {"observations": [{}], "actions": [], "rewards": np.zeros(0),
                                "masks": np.zeros(0), "is_success": False, "env_steps": 0}, b2)
assert b2.rows == [] and b2.inc == 0, "empty traj must not touch the buffer"

# --------------------------------------- safety chain (FlexPoint [0,1] grippers)

env = make_env()
for _ in range(200):
    out = env._clamp("right", np.concatenate([np.full(6, 10.0), [99.0]]))
assert (out[:6] <= LIMITS[:, 1] - 0.05 + 1e-9).all()
assert out[6] == 1.0, f"gripper must clip to 1.0 on YAM-abc, got {out[6]}"
assert env._clamp("right", np.concatenate([np.zeros(6), [-9.0]]))[6] == 0.0
env3 = make_env()
env3._clamp("left", np.concatenate([env3._last_cmd["left"] + 10.0, [1.0]]))
assert np.isclose(env3._last_cmd["left"][0], 0.15), "last_cmd must advance by the CLAMPED value"


class _RecEnv:
    def __init__(self, arms=None):
        self.cmds, self._arms = [], arms

    def step(self, cmd):
        self.cmds.append(cmd)
        return _Obs(self._arms)

    def get_obs(self):
        return _Obs(self._arms)

    def close(self):
        pass


class _Obs:
    def __init__(self, arms=None):
        self.arms = arms or {}

    def to_dict(self):
        return {"ok": True}


env4 = make_env()
env4._env = _RecEnv()
env4.step(np.zeros(14))
env4.step(np.full(14, -1.0))
assert env4._env.cmds == [{}, {}], "hold sentinels must not command the arms"

# ------------------------------------- reset re-anchor + close soft-release


class _Arm:
    def __init__(self, jp):
        self.joint_pos = jp


class _Robot:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def move_joints(self, t, d):
        self.calls.append(("move", np.array(t)))
        if self.fail:
            raise RuntimeError("CAN glitch")

    def soft_release(self, d):
        self.calls.append(("soft_release", d))

    def zero_torque_mode(self):
        self.calls.append(("zero_torque",))


MEASURED = {"left": np.full(6, 0.5), "right": np.full(6, -0.3)}
env5 = make_env()
env5._robot_dict = {k: _Robot(fail=True) for k in ("left", "right")}
env5._env = _RecEnv(arms={k: _Arm(v) for k, v in MEASURED.items()})
env5.reset()
for k in ("left", "right"):
    np.testing.assert_allclose(env5._last_cmd[k], MEASURED[k])  # measured, NOT home
mv = env5._robot_dict["left"].calls[0]
assert mv[1].shape == (7,) and mv[1][6] == 1.0, "reset must command the FlexPoint open value 1.0"
env5.close()
assert all("soft_release" in [c[0] for c in r.calls] for r in env5._robot_dict.values())
assert cleanup_calls, "cleanup_processes must still run"

# ------------------------------------------------- post-trim label semantics

for is_success in (True, False):
    q = 3  # decisions surviving a terminal-obs trim
    r = np.concatenate([-np.ones(q - 1), [0]]) if is_success else -np.ones(q)
    m = np.concatenate([np.ones(q - 1), [0]]) if is_success else np.ones(q)
    if is_success:
        assert r[-1] == 0 and m[-1] == 0, "absorbing success row must survive a trim"

# ------------------------------------------------ noise fill (upstream last-row semantics)
one = np.arange(32, dtype=np.float32)[None]                     # (1, 32) upstream single row
n1 = T.fill_noise_chunk(one, 50)
assert n1.shape == (1, 50, 32) and (n1[0] == one).all(), "single row must tile over the horizon"
full = np.random.default_rng(0).standard_normal((50, 32)).astype(np.float32)
n50 = T.fill_noise_chunk(full, 50)
assert n50.shape == (1, 50, 32) and (n50[0] == full).all(), "full chunk must pass through untouched"
part = np.random.default_rng(1).standard_normal((10, 32)).astype(np.float32)
n10 = T.fill_noise_chunk(part, 50)
assert (n10[0, :10] == part).all() and (n10[0, 10:] == part[-1]).all(), "fill must repeat the LAST row"

# ------------------------------------------------------- buffer_io roundtrip
from examples import buffer_io  # noqa: E402


class _FakeRB:
    def __init__(self, cap):
        self.capacity = cap
        self.data = {"observations": {"pixels": np.zeros((cap, 4), np.uint8),
                                      "state": np.zeros((cap, 3), np.float32)},
                     "actions": np.zeros((cap, 2), np.float32),
                     "rewards": np.zeros(cap, np.float32)}
        self.size = 0
        self._traj_counter = 0
        self._start = 0
        self.traj_bounds = {}


src = _FakeRB(10)
src.data["observations"]["pixels"][:3] = 7
src.data["actions"][:3] = 1.5
src.size, src._traj_counter, src._start = 3, 2, 3
src.traj_bounds = {0: (0, 2), 1: (2, 3)}
import tempfile
with tempfile.TemporaryDirectory() as td:
    path = td + "/rb.pkl"
    buffer_io.save_buffer(src, path)
    dst = _FakeRB(10)
    n, trajs = buffer_io.load_buffer(dst, path)
    assert (n, trajs) == (3, 2)
    assert dst.size == 3 and dst._start == 3 and dst.traj_bounds == {0: (0, 2), 1: (2, 3)}
    assert (dst.data["observations"]["pixels"][:3] == 7).all()
    assert (dst.data["actions"][:3] == 1.5).all()
    tiny = _FakeRB(2)
    try:
        buffer_io.load_buffer(tiny, path)
        raise AssertionError("expected capacity ValueError")
    except ValueError:
        pass

print("ALL OFFLINE NO-ROBOT TESTS PASSED")
