"""YamEnv — limb-backed robot env for the DSRL full-task YAM baseline.

Composes the limb stack in-process (cameras + arms as Portal subprocesses,
RobotEnv pacing the loop at 30 Hz) and exposes the small surface the DSRL
training loop needs: get_observation / step / reset / hold / close.

DSRL stays a *client* of limb: this file imports limb as a library; nothing in
limb or SubRL-VLA is modified. All DROID assumptions from the upstream repo are
gone — actions here are ABSOLUTE joint radians, shape (14,) laid out as
[l_j0..5, l_grip, r_j0..5, r_grip], executed through a safety chain
(hold-guard, per-tick joint delta clamp, joint-limit clamp with margin,
gripper clip to [0, 2.4]) because the i2rt motor chain raises RuntimeError and
kills the robot subprocess on a raw limit violation (port plan B3).

See SubRL-VLA/docs/dsrl_yam_port_plan.md §3b.1 / B3 / B4 / B5 / B9.
"""

import os
import pathlib
import threading
from typing import Any, Dict, Optional

import numpy as np
import yaml

# Trainer hold sentinels (port plan B3): upstream DSRL/EXPO trainers stream
# all-zeros (and some code paths all -1) when there is no plan. Under absolute
# joint control those would command a violent move — hold instead.
_HOLD_SENTINELS = (0.0, -1.0)


class YamEnv:
    """limb RobotEnv wrapper with the DSRL env surface and an action safety chain.

    Parameters
    ----------
    limb_root : str
        Path to the limb repo checkout. The process chdir()s here: limb resolves
        robot_configs/, scripts/reset_all_can.sh, and URDF paths relative to it.
    config_path : str
        Launch YAML (relative to limb_root) whose `robots:` and `sensors:`
        sections define the arms and the three RealSense cameras. The agent/
        collection sections of that YAML are ignored — only hardware is reused.
    control_hz : float
        RobotEnv rate. The env owns the clock (plan B5): step() sleeps to this
        rate internally, so the training loop must NOT sleep on its own.
    joint_delta_limit : float
        Max commanded joint change per tick, radians (per joint, both arms).
    joint_limit_margin : float
        Margin inside the robot_configs joint limits, radians.
    gripper_clip : (float, float)
        Commanded gripper range. Obs gripper is [0, 1]; commands are [0, 2.4].
    gripper_open_cmd : float
        "Open" command used during reset (2.2 per the SubRL configs).
    reset_duration_s : float
        Interpolated move-to-home duration in reset().
    inject_ee_pose : bool
        Fill obs[side]["ee_pose"] via limb's pinocchio FK injector (needed by
        any verifier that reads ee geometry; harmless otherwise).
    """

    def __init__(
        self,
        limb_root: str = "/home/ssc/Desktop/research/limb",
        config_path: str = "configs/yam_subtask_rl_grasp.yaml",
        control_hz: float = 30.0,
        joint_delta_limit: float = 0.15,
        joint_limit_margin: float = 0.05,
        gripper_clip: tuple = (0.0, 2.4),
        gripper_open_cmd: float = 2.2,
        reset_duration_s: float = 4.0,
        inject_ee_pose: bool = True,
        setup_can: bool = True,
    ) -> None:
        self._limb_root = pathlib.Path(limb_root).resolve()
        if not self._limb_root.is_dir():
            raise FileNotFoundError(f"limb_root does not exist: {self._limb_root}")
        # limb resolves robot configs, CAN reset script, and URDFs relative to
        # the repo root — everything else in the DSRL loop uses absolute paths.
        os.chdir(self._limb_root)

        self.control_hz = float(control_hz)
        self._joint_delta_limit = float(joint_delta_limit)
        self._joint_limit_margin = float(joint_limit_margin)
        self._gripper_clip = (float(gripper_clip[0]), float(gripper_clip[1]))
        self.gripper_open_cmd = float(gripper_open_cmd)
        self._reset_duration_s = float(reset_duration_s)

        from omegaconf import OmegaConf

        from limb.envs.robot_env import RobotEnv
        from limb.utils import launch_utils

        launch_utils.setup_logging()
        cfg = OmegaConf.load(self._limb_root / config_path)
        if "robots" not in cfg or "sensors" not in cfg:
            raise ValueError(f"{config_path} must define 'robots' and 'sensors' sections")

        if setup_can:
            launch_utils.setup_can_interfaces()

        self._server_processes: list = []
        camera_dict, _ = launch_utils.initialize_sensors(
            OmegaConf.to_container(cfg.sensors, resolve=True), self._server_processes
        )
        robot_dict = launch_utils.initialize_robots(cfg.robots, self._server_processes)
        self._robot_dict = robot_dict
        self._env = RobotEnv(robot_dict, camera_dict, control_rate_hz=self.control_hz)

        self._joint_limits = self._load_joint_limits(cfg.robots)

        self._ee_injector = None
        if inject_ee_pose:
            try:
                from limb.agents.policy_learning.subtask.fk import EEPoseInjector

                self._ee_injector = EEPoseInjector(sides=["right"])
            except Exception as e:  # pinocchio optional in the dsrl venv
                print(f"[YamEnv] EE-pose FK injection unavailable ({e}); continuing without it")

        # Boot pose = home pose (plan B9: no authoritative reset pose exists in
        # limb; the launch.py boot-pose save is the precedent). Stage the scene
        # with the arms where episodes should start BEFORE launching training.
        first_obs = self._env.reset()
        self._home: Dict[str, np.ndarray] = {}
        for name in ("left", "right"):
            arm = first_obs.arms.get(name)
            if arm is None:
                raise RuntimeError(f"robot '{name}' missing from limb obs — check {config_path} robots section")
            self._home[name] = np.asarray(arm.joint_pos, dtype=np.float64).copy()
        # Delta-clamp reference: last commanded joints (init = measured at boot).
        self._last_cmd = {k: v.copy() for k, v in self._home.items()}
        self._pending_obs: Optional[dict] = None

    # ------------------------------------------------------------------ config

    def _load_joint_limits(self, robots_cfg) -> Dict[str, np.ndarray]:
        """Per-arm (6, 2) joint limits from the layered robot_configs YAMLs (last file wins)."""
        limits: Dict[str, np.ndarray] = {}
        for name, paths in dict(robots_cfg).items():
            merged = None
            for p in list(paths):
                with open(self._limb_root / p) as f:
                    doc = yaml.safe_load(f) or {}
                if "joint_limits" in doc:
                    merged = np.asarray(doc["joint_limits"], dtype=np.float64)
            if merged is None or merged.shape != (6, 2):
                raise ValueError(f"no (6,2) joint_limits found in robot configs for arm '{name}': {list(paths)}")
            limits[name] = merged
        return limits

    # ------------------------------------------------------------------- obs

    def get_observation(self) -> dict:
        """Raw limb obs dict: {left/right: {joint_pos, gripper_pos, ...}, <camera>: {images: {rgb}}, ...}.

        Returns the obs captured by the immediately preceding step() when one is
        pending (RobotEnv.step already reads sensors post-sleep — re-reading
        would double the camera load), otherwise reads fresh.
        """
        if self._pending_obs is not None:
            obs, self._pending_obs = self._pending_obs, None
            return obs
        return self._to_dict(self._env.get_obs())

    def _to_dict(self, obs) -> dict:
        d = obs.to_dict()
        if self._ee_injector is not None:
            try:
                self._ee_injector.inject(d)
            except Exception:
                pass  # FK is best-effort context for verifiers, never load-bearing for control
        return d

    # ------------------------------------------------------------------- act

    def step(self, action: np.ndarray) -> None:
        """Execute one absolute-joint action (14,) through the safety chain at control_hz."""
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (14,):
            raise ValueError(f"YamEnv.step expects a (14,) action, got {a.shape}")
        if any(np.allclose(a, s) for s in _HOLD_SENTINELS):
            self.hold()
            return
        cmd = {
            "left": {"pos": self._clamp("left", a[:7])},
            "right": {"pos": self._clamp("right", a[7:])},
        }
        self._pending_obs = self._to_dict(self._env.step(cmd))

    def _clamp(self, name: str, target7: np.ndarray) -> np.ndarray:
        joints = target7[:6].copy()
        # Per-tick delta clamp vs the last commanded pose, then absolute limits
        # with margin — i2rt raises (and kills the robot subprocess) past the
        # raw limits, so we must never command across them (plan B3).
        last = self._last_cmd[name]
        joints = np.clip(joints, last - self._joint_delta_limit, last + self._joint_delta_limit)
        lim = self._joint_limits[name]
        joints = np.clip(joints, lim[:, 0] + self._joint_limit_margin, lim[:, 1] - self._joint_limit_margin)
        self._last_cmd[name] = joints.copy()
        gripper = np.clip(target7[6], *self._gripper_clip)
        return np.concatenate([joints, [gripper]])

    def hold(self) -> None:
        """One tick at control_hz without commanding the arms (empty action = hold)."""
        self._pending_obs = self._to_dict(self._env.step({}))

    # ------------------------------------------------------------------ reset

    def reset(self) -> dict:
        """Open grippers and safe-move both arms back to the boot home pose.

        Robot motion only — scene re-staging (vial placement) is owned by the
        caller (operator gate in keypress mode; reset policy in a future
        autonomous mode). Returns a fresh obs dict.
        """
        targets = {
            name: np.concatenate([self._home[name], [self.gripper_open_cmd]])
            for name in self._robot_dict
            if name in self._home
        }

        def _move_one(name: str, target: np.ndarray) -> None:
            try:
                self._robot_dict[name].move_joints(target, self._reset_duration_s)
            except Exception as e:
                print(f"[YamEnv] safe-move of '{name}' failed: {e}")

        threads = []
        for name, target in targets.items():
            t = threading.Thread(target=_move_one, args=(name, target), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=self._reset_duration_s + 2.0)
            if t.is_alive():
                print("[YamEnv] safe-move still running after timeout — treating it as failed")

        # Re-anchor the delta clamp to the MEASURED pose, never the assumed home:
        # move_joints can fail or hang (exceptions above are swallowed to keep the
        # run alive), and anchoring to home would let the next step() command a
        # full-distance jump at full PD gains — the exact violent motion the
        # clamp exists to prevent. Anchored to the measured pose, a failed
        # safe-move degrades to a slow delta-limited crawl instead.
        self._pending_obs = None
        obs = self._env.get_obs()
        for name in self._last_cmd:
            measured = np.asarray(obs.arms[name].joint_pos, dtype=np.float64).copy()
            drift = float(np.max(np.abs(measured - self._home[name])))
            if drift > 0.2:
                print(f"[YamEnv] '{name}' is {drift:.2f} rad from home after reset — "
                      "safe-move may have failed; check the arm before continuing")
            self._last_cmd[name] = measured
        return self._to_dict(obs)

    # ------------------------------------------------------------------ misc

    def close(self) -> None:
        from limb.utils.launch_utils import cleanup_processes

        # Ramp the arms down before killing their Portal subprocesses — limb's own
        # shutdown treats soft_release as mandatory (launch.py _safe_release_robots);
        # SIGKILLing a holding arm would skip the gravity-comp fade-out.
        def _release_one(name, robot) -> None:
            try:
                robot.soft_release(2.0)
            except Exception as e:
                print(f"[YamEnv] soft_release('{name}') failed ({e}); falling back to zero_torque_mode")
                try:
                    robot.zero_torque_mode()
                except Exception:
                    pass

        threads = []
        for name, robot in self._robot_dict.items():
            t = threading.Thread(target=_release_one, args=(name, robot), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=5.0)

        try:
            self._env.close()
        except Exception as e:
            print(f"[YamEnv] env.close failed: {e}")
        cleanup_processes(None, self._server_processes)
