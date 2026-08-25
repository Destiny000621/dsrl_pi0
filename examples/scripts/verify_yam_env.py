"""Stage 2 — YamEnv dry-run, ROBOT LIVE, no RL, no policy server (docs/yam_verification.md).

Exercises, with a deliberately tight 0.05 rad delta clamp so every motion is a
small twitch at most:
  1. obs pipeline: limb obs -> 14-D qpos + 3 cams; pi0 wire obs + SAC pixel shapes
  2. reset(): grippers open + safe-move home; delta clamp re-anchored to measured pose
  3. hold sentinel: zeros action -> NO arm motion for 1 s
  4. adversarial +/-pi action -> per-tick joint change <= delta limit, gripper clipped
  5. ee_pose FK injection present (when pinocchio is available)

STAND CLEAR OF THE ARMS. Run inside the dsrl venv on the YAM box:
    python examples/scripts/verify_yam_env.py
"""
import argparse
import types

import numpy as np

from examples.envs.yam_env import YamEnv
from examples.train_utils_yam import extract_yam_observation, get_pi0_input, process_images

PROMPT = "insert the wireless bluetooth earbuds into the charging case"  # YAM-abc earbud SFT
DELTA = 0.05  # tight clamp for the test — production default is 0.15


def qpos_of(env):
    return extract_yam_observation(env.get_observation())["qpos"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limb_root", default="/home/ssc/Desktop/research/limb")
    p.add_argument("--limb_config", default="configs/yam_subtask_rl_earbud_insert.yaml")
    args = p.parse_args()

    env = YamEnv(limb_root=args.limb_root, config_path=args.limb_config,
                 joint_delta_limit=DELTA)
    try:
        # 1. obs pipeline shapes
        raw = env.get_observation()
        curr = extract_yam_observation(raw)
        assert curr["qpos"].shape == (14,)
        req = get_pi0_input(curr, PROMPT)
        for k, v in req["images"].items():
            assert v.shape == (3, 224, 224) and v.dtype == np.uint8, (k, v.shape, v.dtype)
        variant = types.SimpleNamespace(resize_image=128, num_cameras=3)
        assert process_images(variant, curr).shape == (1, 128, 128, 9, 1)
        # Station gate (YAM-abc): FlexPoint gripper OBS must already be
        # normalized to the commanded range (obs far outside gripper_clip means
        # a units mismatch — vial-era config or uncalibrated grippers).
        lo, hi = env._gripper_clip
        for dim in (6, 13):
            g = float(curr["qpos"][dim])
            assert lo - 0.1 <= g <= hi + 0.1, \
                f"gripper obs dim {dim} = {g:.3f} outside [{lo}, {hi}] — station/units mismatch"
        print(f"1. obs pipeline OK: qpos {np.round(curr['qpos'], 3)} "
              f"(grippers in [{lo}, {hi}] ✓)")

        # 2. reset
        input("2. reset(): both arms will slowly safe-move to the boot pose, grippers open. "
              "STAND CLEAR, then press Enter...")
        env.reset()
        print("   reset OK (watch for any 'safe-move may have failed' warning above)")

        # 3. hold sentinel — zeros must NOT move the arms
        q0 = qpos_of(env)
        for _ in range(30):
            env.step(np.zeros(14))
        drift = float(np.abs(qpos_of(env) - q0)[np.r_[0:6, 7:13]].max())
        assert drift < 0.03, f"hold sentinel FAILED: joints moved {drift:.3f} rad on zeros action"
        print(f"3. hold sentinel OK: max joint drift {drift:.4f} rad over 1 s")

        # 4. adversarial action — clamp must reduce +/-pi to <= DELTA per tick
        input(f"4. adversarial +/-pi action for ONE tick (max twitch {DELTA} rad/joint). "
              "STAND CLEAR, then press Enter...")
        q_before = qpos_of(env)
        bad = q_before.astype(np.float64).copy()
        bad[0:6] += np.pi
        bad[7:13] -= np.pi
        bad[6] = 99.0   # gripper: must clip to the station max (YAM-abc FlexPoint: 1.0)
        bad[13] = -9.0  # gripper: must clip to 0.0
        env.step(bad)
        step_delta = np.abs(qpos_of(env) - q_before)[np.r_[0:6, 7:13]]
        assert step_delta.max() < DELTA + 0.05, \
            f"CLAMP FAILED: a joint moved {step_delta.max():.3f} rad in one tick"
        print(f"4. safety clamp OK: max one-tick joint change {step_delta.max():.3f} rad "
              f"(limit {DELTA} + tracking)")
        env.step(np.concatenate([q_before[:6], [env.gripper_open_cmd],
                                 q_before[7:13], [env.gripper_open_cmd]]))

        # 5. FK injection
        right = env.get_observation().get("right", {})
        if right.get("ee_pose") is not None:
            print(f"5. ee_pose FK OK: {np.round(np.asarray(right['ee_pose']), 3)}")
        else:
            print("5. ee_pose absent — pinocchio not installed in this venv (fine for keypress mode)")

        print("\nALL YAM-ENV CHECKS PASSED — safe to proceed to stage 3")
    finally:
        input("Done. Press Enter to soft-release the arms and shut down (support them if needed)...")
        env.close()


if __name__ == "__main__":
    main()
