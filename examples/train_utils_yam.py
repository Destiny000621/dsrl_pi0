"""DSRL training loop for the YAM full-task baseline — clone of train_utils_real.py
with the DROID-specific plumbing replaced (port plan §3b.3, blockers B1-B5, B10b).

Differences vs train_utils_real.py, each tied to a plan item:
  B2  noise is tiled to the model's action_horizon (50), not a hardcoded 10;
  B3  NO gripper binarization / [-1,1] clip — actions are absolute joint radians,
      safety (delta clamp, joint limits, gripper clip) lives in YamEnv.step;
  B4  obs extraction reads the limb obs dict (3 RGB RealSense cams, 14-D qpos)
      and builds the pi0.5 ALOHA wire obs (CHW uint8 224, cam_high/left/right);
  B5  no client-side sleep — YamEnv/RobotEnv own the 30 Hz clock;
  B8  get_prefix_rep returns {"prefix_rep": [1, emb]} = the OFFICIAL last-
      prefix-slot feature (upstream's hidden_state[:, -1, :], sliced server-
      side), not the fork's full (hidden_state, kv_cache) tuple;
  B10b is_success is initialized before the episode body.

Reward convention is unchanged from upstream DSRL (per-decision -1/0, success =
absorbing goal with mask 0; failure/timeout non-terminal with mask 1).
wandb additionally logs the SubRL parity metrics episode_reward /
success_rate_10 / episode_steps (plan §6 step 7).
"""

import os
import select
import sys
import termios
import time
import tty

import jax
import numpy as np
from moviepy.editor import ImageSequenceClip
from openpi_client import image_tools
from tqdm import tqdm

# pi0.5 ALOHA wire names (SFT ground truth, plan §1): limb camera name -> server image name.
CAMERA_TO_WIRE = {
    "head_camera": "cam_high",
    "left_wrist_camera": "cam_left_wrist",
    "right_wrist_camera": "cam_right_wrist",
}
# SAC pixel concat order (deterministic, mirrors process_images in train_utils_real).
SAC_CAMERA_ORDER = ("head_camera", "left_wrist_camera", "right_wrist_camera")
VIDEO_CAMERA = "head_camera"


def trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer,
                                       replay_buffer, wandb_logger, shard_fn=None, agent_dp=None):
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    i = 0
    total_env_steps = 0
    total_num_traj = 0
    episode_rewards = []  # SubRL parity: rolling success_rate_10 window
    # A restored SAC actor should act (and skip the 5000-step warmup block) from
    # the first episode; the i==0 N(0,1) phase is for FRESH runs only.
    fresh_start = getattr(variant, 'restore_path', '') == ''
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)

    max_episodes = getattr(variant, 'max_episodes', -1)
    with tqdm(total=variant.max_steps, initial=0) as pbar:
        while i <= variant.max_steps:
            if max_episodes > 0 and total_num_traj >= max_episodes:
                print(f'Episode budget reached ({total_num_traj}/{max_episodes}) — stopping.')
                break
            traj = collect_traj(variant, agent, env, i, agent_dp, wandb_logger, total_num_traj)
            total_num_traj += 1
            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            total_env_steps += traj['env_steps']
            print('online buffer timesteps length:', len(online_replay_buffer))
            print('online buffer num traj:', total_num_traj)
            print('total env steps:', total_env_steps)

            # SubRL parity metrics (plan §6.7): episode_reward is the 0/1 outcome,
            # success_rate_10 only once a full 10-episode window exists (the
            # partial-window mean swings the early curve wildly).
            episode_rewards.append(float(traj['is_success']))
            parity = {
                'episode_reward': episode_rewards[-1],
                'episode_steps': int(traj['env_steps']),
                'episodes_seen': len(episode_rewards),
            }
            if len(episode_rewards) >= 10:
                parity['success_rate_10'] = float(np.mean(episode_rewards[-10:]))
            wandb_logger.log(parity, step=i)

            if i == 0 and fresh_start:
                num_gradsteps = 5000
            else:
                num_gradsteps = len(traj["rewards"]) * variant.multi_grad_step
            print(f'num_gradsteps: {num_gradsteps}')
            # len() guard: an empty first trajectory (aborted episode) would make
            # ReplayBuffer.sample crash on randint(0, 0).
            if total_num_traj >= variant.num_initial_traj_collect and len(online_replay_buffer) > 0:
                for _ in range(num_gradsteps):

                    batch = next(replay_buffer_iterator)
                    update_info = agent.update(batch)

                    pbar.update()
                    i += 1

                    if i % variant.log_interval == 0:
                        update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                        for k, v in update_info.items():
                            if v.ndim == 0:
                                wandb_logger.log({f'training/{k}': v}, step=i)
                            elif v.ndim <= 2:
                                wandb_logger.log_histogram(f'training/{k}', v, i)
                        wandb_logger.log({
                            'replay_buffer_size': len(online_replay_buffer),
                            'is_success (exploration)': int(traj['is_success']),
                        }, i)

                    if i % variant.eval_interval == 0:
                        wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                        wandb_logger.log({'num_online_trajs': total_num_traj}, step=i)
                        wandb_logger.log({'env_steps': total_env_steps}, step=i)
                        if hasattr(agent, 'perform_eval'):
                            try:
                                agent.perform_eval(variant, i, wandb_logger, replay_buffer,
                                                   replay_buffer_iterator, eval_env)
                            except Exception as e:
                                # Upstream's Q-visualization asserts 3-channel pixels;
                                # ours are 9-channel (3 cams). A viz failure must never
                                # kill a multi-hour robot run.
                                print(f'perform_eval skipped: {e}')

                    if variant.checkpoint_interval != -1:
                        if i % variant.checkpoint_interval == 0:
                            agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)


def add_online_data_to_buffer(variant, traj, online_replay_buffer):
    # Unchanged from train_utils_real.py: one buffer row per noise DECISION;
    # the effective per-decision discount is discount ** query_freq.
    discount_horizon = variant.query_freq
    actions = np.array(traj['actions'])
    episode_len = len(actions)
    if episode_len == 0:
        print('Empty trajectory — skipping buffer add.')
        return
    rewards = np.array(traj['rewards'])
    masks = np.array(traj['masks'])

    for t in range(episode_len):
        obs = traj['observations'][t]
        next_obs = traj['observations'][t + 1]
        # remove batch dimension
        obs = {k: v[0] for k, v in obs.items()}
        next_obs = {k: v[0] for k, v in next_obs.items()}
        if not variant.add_states:
            obs.pop('state', None)
            next_obs.pop('state', None)

        insert_dict = dict(
            observations=obs,
            next_observations=next_obs,
            actions=actions[t],
            next_actions=actions[t + 1] if t < episode_len - 1 else actions[t],
            rewards=rewards[t],
            masks=masks[t],
            discount=variant.discount ** discount_horizon
        )
        online_replay_buffer.insert(insert_dict)
    online_replay_buffer.increment_traj_counter()


# --------------------------------------------------------------------- obs

def extract_yam_observation(raw_obs: dict) -> dict:
    """limb obs dict -> {<camera>_image (HWC uint8 RGB), qpos (14,) float32}.

    qpos layout matches the SFT state exactly (plan §1):
    [left.joint_pos(6), left.gripper_pos(1), right.joint_pos(6), right.gripper_pos(1)].
    limb RealSense frames are already RGB uint8 — no BGR flip, no alpha (B4).
    """
    out = {}
    for cam in CAMERA_TO_WIRE:
        try:
            img = raw_obs[cam]["images"]["rgb"]
        except (KeyError, TypeError) as e:
            raise KeyError(f"camera '{cam}' missing from limb obs (keys: {list(raw_obs)})") from e
        img = np.asarray(img)
        if img.dtype != np.uint8:
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
        out[f"{cam}_image"] = img
    qpos = np.concatenate([
        np.asarray(raw_obs["left"]["joint_pos"], dtype=np.float32).reshape(-1),
        np.asarray(raw_obs["left"]["gripper_pos"], dtype=np.float32).reshape(-1),
        np.asarray(raw_obs["right"]["joint_pos"], dtype=np.float32).reshape(-1),
        np.asarray(raw_obs["right"]["gripper_pos"], dtype=np.float32).reshape(-1),
    ])
    if qpos.shape != (14,):
        raise ValueError(f"expected 14-D qpos, got {qpos.shape}")
    out["qpos"] = qpos
    return out


def get_pi0_input(curr_obs: dict, instruction: str) -> dict:
    """pi0.5 ALOHA wire obs (plan §1): CHW uint8 224x224 pad-resize, cam_high/left/right names.

    Client-side pre-resize is safe (ResizeImages is idempotent server-side) and cuts
    the payload ~6x. CHW is mandatory — the server's _decode_aloha rearranges c h w -> h w c
    unconditionally, so HWC would silently scramble the image.
    """
    images = {}
    for cam, wire_name in CAMERA_TO_WIRE.items():
        img = image_tools.resize_with_pad(curr_obs[f"{cam}_image"], 224, 224)
        images[wire_name] = np.transpose(img, (2, 0, 1))
    return {
        "state": curr_obs["qpos"],
        "images": images,
        "prompt": instruction,
    }


def process_images(variant, curr_obs: dict) -> np.ndarray:
    """SAC pixels: 3 cams pad-resized to resize_image, channel-concat -> (1, H, W, 9, 1)."""
    ims = [
        image_tools.resize_with_pad(curr_obs[f"{cam}_image"], variant.resize_image, variant.resize_image)
        for cam in SAC_CAMERA_ORDER
    ]
    return np.concatenate(ims, axis=2)[np.newaxis, ..., np.newaxis]


def get_sac_obs(variant, curr_obs: dict, agent_dp, request_data: dict) -> dict:
    """SAC obs dict: pixels + [qpos(14), last-prefix-slot z(emb)] state (plan §1 'SAC obs', B8)."""
    img_all = process_images(variant, curr_obs)
    prefix_rep = np.asarray(agent_dp.get_prefix_rep(request_data)["prefix_rep"], dtype=np.float32)
    state = np.concatenate([curr_obs["qpos"], prefix_rep.flatten()])
    return {
        'pixels': img_all,
        'state': state[np.newaxis, ..., np.newaxis],
    }


# ---------------------------------------------------------------- keyboard

def _drain_stdin() -> None:
    while select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
        sys.stdin.read(1)


def _wait_for_key(valid: str, prompt: str) -> str:
    """Block (cbreak assumed active) until one of `valid` chars is pressed."""
    print(prompt, flush=True)
    while True:
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            ch = sys.stdin.read(1).lower()
            if ch in valid:
                return ch
            print(f"Invalid input {ch!r}. Expected one of: {'/'.join(valid)}", flush=True)
        time.sleep(0.01)


# ------------------------------------------------------------------- traj

def collect_traj(variant, agent, env, i, agent_dp=None, wandb_logger=None, traj_id=None):
    query_frequency = variant.query_freq
    action_horizon = variant.action_horizon  # 50 for pi0.5 (B2 — never hardcode 10)
    noise_dim = variant.noise_dim
    instruction = variant.instruction
    max_timesteps = variant.max_timesteps
    agent._rng, rng = jax.random.split(agent._rng)
    is_success = False  # B10c: defined before any code that can raise

    rewards = []
    action_list = []
    obs_list = []
    image_list = []

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())

        # Robot to home (B9 — YamEnv.reset moves; scene staging is the operator's).
        env.reset()
        _drain_stdin()
        _wait_for_key('c', "Stage the scene for a new episode, then press 'c' to start "
                           "(episode runs %d steps @ %.0f Hz; 'q' aborts)." %
                           (max_timesteps, getattr(env, 'control_hz', 30.0)))

        action = None
        t = 0
        env_steps_executed = 0
        try:
            for t in tqdm(range(max_timesteps)):
                # 'q' aborts the episode early; the outcome is still labeled below.
                if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                    if sys.stdin.read(1).lower() == 'q':
                        print("'q' pressed, stopping episode.")
                        break

                raw_obs = env.get_observation()
                curr_obs = extract_yam_observation(raw_obs)
                image_list.append(curr_obs[f"{VIDEO_CAMERA}_image"])
                request_data = get_pi0_input(curr_obs, instruction)

                if t % query_frequency == 0:
                    rng, key = jax.random.split(rng)
                    obs_dict = get_sac_obs(variant, curr_obs, agent_dp, request_data)

                    if i == 0 and not getattr(variant, 'restore_path', ''):
                        # Base-policy phase (fresh runs only — a restored actor acts
                        # immediately): ONE N(0,1) row tiled across the horizon —
                        # the same single-row-tiled structure the SAC produces, so the
                        # first 5000-grad-step block trains on in-distribution actions.
                        noise_row = jax.random.normal(key, (1, 1, noise_dim))
                        noise = np.asarray(jax.numpy.repeat(noise_row, action_horizon, axis=1))
                        actions_noise = np.asarray(noise_row[0])  # (1, noise_dim) — the SAC action
                    else:
                        actions_noise = agent.sample_actions(obs_dict)
                        actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)  # (1, noise_dim)
                        noise = np.repeat(actions_noise, action_horizon, axis=0)[None]  # (1, H, noise_dim)
                    action_list.append(actions_noise)
                    obs_list.append(obs_dict)
                    action = agent_dp.infer(request_data, noise=np.asarray(noise))["actions"]

                # Absolute joint action; safety chain (delta clamp, joint limits,
                # gripper clip, hold sentinels) lives in YamEnv.step (B3). No
                # client-side sleep — the env paces at control_hz (B5).
                env.step(np.asarray(action[t % query_frequency]))
                env_steps_executed += 1
        except Exception:
            # A transient hardware/server failure must not kill a multi-hour run:
            # end the episode here, let the operator label it, keep training.
            import traceback
            traceback.print_exc()
            print("Episode errored mid-rollout — label the outcome and re-stage.")

        _drain_stdin()
        ch = _wait_for_key('10', "Episode finished. Mark outcome: '1' = SUCCESS, '0' = FAILURE.")
        is_success = ch == '1'
        print(f"Trial marked as {'SUCCESS' if is_success else 'FAILURE'}.")

        try:
            # Terminal observation (same replan-time treatment as upstream). If this
            # fails too, the post-finally trim drops the last decision so the traj
            # stays (obs, action, next_obs)-consistent.
            raw_obs = env.get_observation()
            curr_obs = extract_yam_observation(raw_obs)
            image_list.append(curr_obs[f"{VIDEO_CAMERA}_image"])
            request_data = get_pi0_input(curr_obs, instruction)
            obs_list.append(get_sac_obs(variant, curr_obs, agent_dp, request_data))
        except Exception:
            import traceback
            traceback.print_exc()
            print("Terminal observation failed — trimming the last decision from the trajectory.")
        print('Rollout Done')

    finally:
        if wandb_logger is not None:
            wandb_logger.log({'is_success': int(is_success)}, step=i)
            wandb_logger.log({'total_num_traj': traj_id}, step=i)

        if image_list:
            video_path = os.path.join(variant.outputdir, f'video_high_{traj_id}.mp4')
            fps = int(getattr(env, 'control_hz', 30.0))
            ImageSequenceClip(list(np.stack(image_list)), fps=fps).write_videofile(video_path, codec="libx264")

        # Back to a safe pose right away; the next episode's start gate handles
        # scene re-staging (replaces the DROID pdb.set_trace stop, B9).
        try:
            env.reset()
        except Exception:
            import traceback
            traceback.print_exc()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    if len(obs_list) != len(action_list) + 1:
        # The episode died before the terminal obs was appended (exception mid-
        # rollout). Trim to a consistent (obs, action, next_obs) set.
        n = max(len(obs_list) - 1, 0)
        print(f"Trimming trajectory from {len(action_list)} to {n} decisions (terminal obs missing).")
        action_list = action_list[:n]

    # Labels are built from the FINAL decision count, AFTER the trim — slicing
    # precomputed labels would delete exactly the reward-0/mask-0 success row
    # and silently store a successful episode as a non-terminal failure.
    # Upstream DSRL convention: -1 per decision, 0 at the absorbing success
    # step (mask 0); failure/timeout is non-terminal (mask 1).
    query_steps = len(action_list)
    if query_steps > 0:
        if is_success:
            rewards = np.concatenate([-np.ones(query_steps - 1), [0]])
            masks = np.concatenate([np.ones(query_steps - 1), [0]])
        else:
            rewards = -np.ones(query_steps)
            masks = np.ones(query_steps)
    else:
        rewards = np.zeros(0)
        masks = np.zeros(0)

    traj = {
        'observations': obs_list,
        'actions': action_list,
        'rewards': rewards,
        'masks': masks,
        'is_success': is_success,
        'env_steps': env_steps_executed,
    }

    return traj
