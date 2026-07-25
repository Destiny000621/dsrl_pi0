#! /usr/bin/env python
"""DSRL on YAM (full-task baseline) — clone of train_real.py with the DROID env
replaced by the limb-backed YamEnv and the B1/B10a fixes from
SubRL-VLA/docs/dsrl_yam_port_plan.md.

Topology (plan §0): this process runs the SAC learner + robot env on the YAM
box; the frozen pi0.5 SFT serves on the same box at :8111 (limb/openpi
serve_policy with the DSRL noise envelope). remote_host/remote_port env vars
override the default 127.0.0.1:8111.
"""
import logging
import os
import tempfile
from functools import partial

import gymnasium as gym
import jax
import numpy as np
import tensorflow as tf
from gym.spaces import Box, Dict
from jax.experimental.compilation_cache import compilation_cache

from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.general_utils import add_batch_dim
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
from openpi_client import websocket_client_policy as _websocket_client_policy

from examples.envs.yam_env import YamEnv
from examples.train_utils_yam import (
    extract_yam_observation,
    get_pi0_input,
    trajwise_alternating_training_loop,
)

home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))


def shard_batch(batch, sharding):
    """Shards a batch across devices along its first dimension."""
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )


class DummyEnv(gym.ObservationWrapper):
    """Spec-only env for agent/buffer construction — never stepped.

    state_dim is MEASURED at startup (14-D qpos + live prefix_rep width) instead
    of the upstream `8 + 2024` constant, which was wrong even for DROID (plan B1).
    """

    def __init__(self, variant, state_dim):
        self.variant = variant
        self.image_shape = (variant.resize_image, variant.resize_image, 3 * variant.num_cameras, 1)
        obs_dict = {}
        obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        if variant.add_states:
            obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        # The SAC action is ONE noise row, tiled across the model's action horizon
        # at replan time — a (1, 32) space, not (horizon, 32) (plan §1 'SAC action').
        self.action_space = Box(low=-1, high=1, shape=(1, variant.noise_dim), dtype=np.float32)


def main(variant):
    if variant.mode != 'keypress':
        raise NotImplementedError(
            f"mode={variant.mode!r}: only 'keypress' (operator reward + operator re-stage) is "
            "implemented. 'autonomous' needs the full-task insertion verifier (plan §5.3b) and "
            "the hybrid reset (§5.4); 'pedal_safety' is phase 2. Run keypress bring-up first."
        )

    devices = jax.local_devices()
    num_devices = len(devices)
    assert variant.batch_size % num_devices == 0
    logging.info('num devices %s', num_devices)
    logging.info('batch size %s', variant.batch_size)
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps

    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)

    # abspath BEFORE YamEnv chdirs into limb_root — relative paths would
    # otherwise silently resolve under the limb repo.
    outputdir = os.path.abspath(os.path.join(os.environ['EXP'], expname))
    variant.outputdir = outputdir
    if variant.restore_path:
        variant.restore_path = os.path.abspath(variant.restore_path)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    print('writing to output dir ', outputdir)

    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(variant.prefix != '', variant, variant.wandb_project,
                               experiment_id=expname, output_dir=wandb_output_dir, group_name=group_name)

    logging.info("initializing the YAM environment (limb in-process)...")
    env = YamEnv(
        limb_root=variant.limb_root,
        config_path=variant.limb_config,
        control_hz=variant.control_hz,
        joint_delta_limit=variant.joint_delta_limit,
    )
    eval_env = env

    agent_dp = _websocket_client_policy.WebsocketClientPolicy(
        host=os.environ.get('remote_host', '127.0.0.1'),
        port=int(os.environ.get('remote_port', '8111')),
    )
    logging.info(f"Server metadata: {agent_dp.get_server_metadata()}")

    try:
        # B1: measure the SAC state width live — 14-D qpos + the served model's
        # pooled prefix_rep — never trust a constant. Also warms the server's
        # embed JIT before the first episode.
        curr_obs = extract_yam_observation(env.get_observation())
        prefix_rep = np.asarray(
            agent_dp.get_prefix_rep(get_pi0_input(curr_obs, variant.instruction))["prefix_rep"]
        )
        state_dim = int(curr_obs["qpos"].size + prefix_rep.size)
        print(f"measured prefix_rep shape {prefix_rep.shape} -> SAC state_dim {state_dim}")

        dummy_env = DummyEnv(variant, state_dim)
        sample_obs = add_batch_dim(dummy_env.observation_space.sample())
        sample_action = add_batch_dim(dummy_env.action_space.sample())
        logging.info('sample obs shapes %s', [(k, v.shape) for k, v in sample_obs.items()])
        logging.info('sample action shape %s', sample_action.shape)

        agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **kwargs)

        if variant.restore_path != '':
            logging.info('restoring from %s', variant.restore_path)
            agent.restore_checkpoint(variant.restore_path)

        online_buffer_size = 2 * variant.max_steps // variant.multi_grad_step
        online_replay_buffer = ReplayBuffer(dummy_env.observation_space, dummy_env.action_space,
                                            int(online_buffer_size))
        replay_buffer = online_replay_buffer
        replay_buffer.seed(variant.seed)

        trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer,
                                           replay_buffer, wandb_logger, shard_fn=shard_fn, agent_dp=agent_dp)
    finally:
        env.close()
