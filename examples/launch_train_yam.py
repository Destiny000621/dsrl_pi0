"""Launcher for DSRL on YAM — clone of launch_train_real.py with the YAM flags
(plan §3b.4): --restore_path (upstream referenced it without defining it, B10a),
--query_freq 25, --action_horizon 50, --max_timesteps 3600 (120 s), --mode.

Hyperparameter defaults follow the real-DROID recipe (plan §5.1 candidate (i));
run_yam.sh overrides discount to 0.999 for the 48-decision full-task horizon.
"""
import argparse
import pathlib
import sys

# Make the repo root importable when run as `python examples/launch_train_yam.py`
# (sys.path[0] is examples/, not the cwd; upstream's launchers share this gap).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from examples.train_yam import main  # noqa: E402
from jaxrl2.utils.launch_util import parse_training_args  # noqa: E402

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, help='Random seed.', type=int)
    parser.add_argument('--launch_group_id', default='dsrl-fulltask', help='group id used to group runs on wandb.')
    parser.add_argument('--eval_episodes', default=10, help='Number of episodes used for evaluation.', type=int)
    parser.add_argument('--env', default='yam', help='name of environment')
    parser.add_argument('--log_interval', default=100, help='Logging interval.', type=int)
    parser.add_argument('--eval_interval', default=2000, help='Eval interval.', type=int)
    parser.add_argument('--checkpoint_interval', default=-1, help='checkpoint interval.', type=int)
    parser.add_argument('--batch_size', default=256, help='Mini batch size.', type=int)
    parser.add_argument('--max_steps', default=int(5e5), help='Number of training (gradient) steps.', type=int)
    parser.add_argument('--add_states', default=1, help='whether to add low-dim states to the observations', type=int)
    parser.add_argument('--wandb_project', default='subrl-yam-earbud-insert',
                        help='wandb project (per-task convention; earbud = subrl-yam-earbud-insert)')
    parser.add_argument('--num_initial_traj_collect', default=1,
                        help='number of trajectories to collect before starting online updates', type=int)
    parser.add_argument('--algorithm', default='pixel_sac', help='type of algorithm')
    parser.add_argument('--prefix', default='', help='prefix to use for wandb')
    parser.add_argument('--suffix', default='', help='suffix to use for wandb')
    parser.add_argument('--multi_grad_step', default=30,
                        help='Number of gradient steps to take per environment step, aka UTD', type=int)
    parser.add_argument('--resize_image', default=128, help='the size of image if need resizing', type=int)
    parser.add_argument('--query_freq', default=25,
                        help='env steps executed open-loop per noise decision (0.83 s @ 30 Hz)', type=int)
    parser.add_argument('--instruction', default='insert the wireless bluetooth earbuds into the charging case',
                        help='language instruction — MUST equal the served SFT default_prompt verbatim '
                             '(YAM-abc earbud = pi05_yam_abc_earbuds; no trailing period)')
    parser.add_argument('--restore_path', default='', help='SAC checkpoint dir to restore from (B10a)')
    parser.add_argument('--action_horizon', default=50,
                        help='served model action horizon the noise row is tiled to (pi0.5 = 50, B2)', type=int)
    parser.add_argument('--noise_dim', default=32,
                        help='flow latent dim per chunk row (pi0.5 padded action_dim = 32)', type=int)
    parser.add_argument('--max_timesteps', default=3600,
                        help='episode step budget (120 s @ 30 Hz; user raised from 40 s on 2026-08-26)', type=int)
    parser.add_argument('--mode', default='keypress', choices=['keypress', 'autonomous', 'pedal_safety'],
                        help="episode reward/reset mode; only 'keypress' is implemented (plan §4)")
    parser.add_argument('--control_hz', default=30.0, help='robot control rate (SFT data rate)', type=float)
    parser.add_argument('--limb_root', default='/home/ssc/Desktop/research/limb', help='limb repo checkout')
    parser.add_argument('--limb_config', default='configs/yam_subtask_rl_earbud_insert.yaml',
                        help='limb launch YAML whose robots/sensors sections define the hardware '
                             '(YAM-abc earbud default; never mix stations)')
    parser.add_argument('--joint_delta_limit', default=0.15,
                        help='max commanded joint change per tick, rad (safety chain, B3)', type=float)
    parser.add_argument('--gripper_open_cmd', default=1.0,
                        help='gripper open command (YAM-abc FlexPoint normalized: 1.0; vial-era was 2.2)',
                        type=float)
    parser.add_argument('--gripper_clip_max', default=1.0,
                        help='max gripper command (YAM-abc FlexPoint: 1.0; vial-era was 2.4)', type=float)

    # The hyperparameters for the real robot experiments (identical to
    # launch_train_real.py; --discount etc. are CLI-overridable via parse_training_args)
    train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        hidden_dims=(1024, 1024, 1024),
        cnn_features=(32, 32, 32, 32),
        cnn_strides=(3, 2, 2, 2),
        cnn_padding='VALID',
        latent_dim=50,
        discount=0.99,
        tau=0.005,
        critic_reduction='min',
        dropout_rate=0.0,
        aug_next=1,
        use_bottleneck=True,
        encoder_type='small',
        encoder_norm='group',
        use_spatial_softmax=True,
        softmax_temperature=-1,
        target_entropy=0.0,
        num_qs=2,
        action_magnitude=2.5,
        num_cameras=3,
    )

    variant, args = parse_training_args(train_args_dict, parser)
    print(variant)
    main(variant)
    sys.exit()
