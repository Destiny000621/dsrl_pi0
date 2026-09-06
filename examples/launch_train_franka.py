#!/usr/bin/env python3
"""Launcher for the Franka DSRL learner — hyperparameters for a 100-EPISODE budget.

Entry point (mirrors upstream's `launch_train_real.py` -> `train_real.py`):

    python -m examples.launch_train_franka --num_episodes 100

Two upstream recipes exist and they are NOT interchangeable. Our pi0.5-SFT is the
**aloha** lineage (action_horizon 50, absolute chunked actions), not pi0-DROID
(horizon 10, joint velocity). So shapes and cadence come from aloha
(`run_aloha.sh`, `train_utils_sim.py`) while the episode/reward/keypress structure
comes from the real-robot script. Where the two disagree on a capacity knob, the
real recipe wins: aloha is a 3M-step SIM run, we have 100 real episodes.

Deviations from both upstream recipes, with the reason (everything else is copied):

  noise_rows 50   MEASURED, not chosen. Chunk-vs-expert position MAE by rows:
                  1 -> 40.7mm, 2 -> 39.5, 5 -> 34.6, 10 -> 31.7, 25 -> 13.0,
                  50 -> 3.3 (server-drawn baseline 4.0). Upstream's (1,32) action
                  space puts the frozen policy 6.7x off its SFT manifold BEFORE any
                  gradient step -- the same magnitude as the 12-44mm insertion error
                  the wcrop retrain exists to fix. `openpi scripts/dsrl_verify_noise.py
                  --rows-sweep`. jaxrl2 flattens 50x32 -> 1600 internally, so the
                  learner is untouched.
  state_dim 2058  MEASURED (10 proprio + 2048 z_rl). Upstream's hardcoded
                  `8 + 2024` is wrong twice over and would be a shape error at
                  the first insert.
  num_cameras 2   Franka has wrist + side. Both upstream recipes assume 3.
  discount .9995  Stored per-decision as discount**query_freq = 0.9995^50 = 0.975,
                  i.e. an effective horizon of ~40 decisions -- about one episode.
                  aloha's 0.999 gives 0.951/decision = horizon 20, half an episode,
                  which leaves the early decisions nearly blind to the only reward
                  that exists. This is the one knob changed for judgement, not
                  measurement; revert to 0.999 if the critic is unstable.
  n_init_traj 5   Upstream collects 1 episode before the first update. At a
                  100-episode budget those first episodes are also the frozen-pi0.5
                  BASELINE the whole run is judged against (they run pure N(0,1)
                  noise), and one episode is too thin to seed 5000 grad steps.
                  5 episodes = 5% of the budget for a real baseline row.
  max_steps 200k  Ceiling, not a target: 100 eps x <=54 decisions x UTD 30 = 162k.
                  It also sizes the buffer (2*max_steps//UTD = 13,333 transitions,
                  ~3 GB) comfortably above the ~4,000 this run will collect, so the
                  buffer never hits upstream's 2x-permanent / 3x-transient resize.
"""

from __future__ import annotations

import argparse
import os
import sys


def build_variant(parser: argparse.ArgumentParser):
    from jaxrl2.utils.launch_util import parse_training_args  # noqa: PLC0415

    # ---- loop / budget ----------------------------------------------------
    parser.add_argument("--num_episodes", default=100, type=int,
                        help="online RL episode budget (bookkeeping + logging only; "
                             "the operator ends the session)")
    parser.add_argument("--port", default=9111, type=int, help="learner service port")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--query_freq", default=50, type=int,
                        help="env steps executed per noise decision. 50 = the full pi0.5 "
                             "chunk, matching BOTH upstream recipes (query_freq == "
                             "action_horizon). 1.67 s per decision at 30 Hz.")
    parser.add_argument("--noise_rows", default=50, type=int,
                        help="rows of the (rows, 32) latent the SAC actor emits. MUST be "
                             "50 on this checkpoint -- see the module docstring.")
    parser.add_argument("--state_dim", default=2058, type=int,
                        help="proprio(10) + z_rl(2048), MEASURED. The service hard-fails "
                             "if the robot sends a different width.")
    parser.add_argument("--warmup_grad_steps", default=5000, type=int,
                        help="grad steps in the FIRST update block (upstream hardcodes "
                             "5000 at train_utils_real.py:38-43); a flag only so the "
                             "smoke test can run a short one")
    parser.add_argument("--num_initial_traj_collect", default=5, type=int,
                        help="episodes collected under N(0,1) before the first update; "
                             "these are the frozen-pi0.5 baseline row")
    parser.add_argument("--eval", default=0, type=int,
                        help="EVALUATION mode: the trained actor drives every decision "
                             "(no N(0,1) warmup), and nothing is learned or stored -- no "
                             "buffer inserts, no gradient steps, no saves. Requires "
                             "--restore_path. Robot side is unchanged: 1/0 still label "
                             "the outcome, which is what gets counted.")
    parser.add_argument("--eval_deterministic", default=0, type=int,
                        help="eval only: use the actor MEAN instead of sampling. Upstream "
                             "DSRL never evaluates deterministically (rollouts and evals "
                             "both sample, jaxrl2/agents/agent.py), so 0 is the "
                             "convention-faithful default; 1 answers a different question "
                             "(the policy's mode, not its behaviour).")

    # ---- bookkeeping ------------------------------------------------------
    parser.add_argument("--prefix", default="dsrl_franka_cable", type=str)
    parser.add_argument("--wandb_project", default="dsrl-franka-cable", type=str,
                        help="empty string disables wandb (the run is then not comparable)")
    parser.add_argument("--wandb_group", default="dsrl-fulltask", type=str)
    parser.add_argument("--outputdir", default="", type=str)
    parser.add_argument("--restore_path", default="", type=str,
                        help="resume actor/critic across sessions (upstream references "
                             "this but never adds the flag -- B10a)")
    parser.add_argument("--restore_buffer", default="", type=str,
                        help="path to a replay_buffer.pkl written by a previous session. "
                             "Upstream never persists the buffer on the real-robot path; "
                             "at 100 episodes those transitions are robot time and cannot "
                             "be regenerated, so resume the buffer too or the resumed run "
                             "is not a continuation")
    parser.add_argument("--save_every_episodes", default=10, type=int,
                        help="checkpoint actor/critic + buffer every N episodes (0 = only "
                             "on shutdown). SIGINT/SIGTERM always saves first")
    parser.add_argument("--log_interval", default=100, type=int)
    parser.add_argument("--checkpoint_interval", default=10000, type=int)
    parser.add_argument("--max_steps", default=200_000, type=int)
    parser.add_argument("--multi_grad_step", default=30, type=int, help="UTD ratio")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--resize_image", default=128, type=int)

    # ---- PixelSACLearner kwargs (real-robot recipe unless noted) ----------
    train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        hidden_dims=(1024,),      # expands to (1024,)*3; aloha's 128 would bottleneck
        cnn_features=(32, 32, 32, 32),
        cnn_strides=(3, 2, 2, 2),
        cnn_padding="VALID",
        latent_dim=50,
        discount=0.9995,
        tau=0.005,
        critic_reduction="min",
        dropout_rate=0.0,
        aug_next=1,
        use_bottleneck=True,
        encoder_type="small",
        encoder_norm="group",
        use_spatial_softmax=True,
        softmax_temperature=-1,
        target_entropy=0.0,
        num_qs=2,
        action_magnitude=2.0,     # aloha lineage; see docs for the measured sweep
        num_cameras=2,            # Franka: wrist + side (upstream assumes 3)
    )
    # NOTE: every key of train_args_dict becomes a --flag via parse_training_args, so
    # none of them may also be declared above (argparse raises on a duplicate option).
    # `--discount` therefore lives here only; the service reads variant.discount and
    # the learner reads train_kwargs["discount"] -- the same value by construction.
    variant, _ = parse_training_args(train_args_dict, parser)
    if not variant.outputdir:
        variant["outputdir"] = os.path.join(
            os.environ.get("EXP", "./logs/dsrl_franka"), f"{variant.prefix}_seed{variant.seed}"
        )
    # flax's checkpoints.save_checkpoint refuses relative paths ("Checkpoint path
    # should be absolute") -- with a relative EXP the buffer saved but the WEIGHTS
    # silently did not (live 2026-09-05: warning per save, zero checkpoints on disk).
    variant["outputdir"] = os.path.abspath(variant.outputdir)
    if variant.restore_path:
        variant["restore_path"] = os.path.abspath(variant.restore_path)
    if variant.eval and not variant.restore_path:
        raise SystemExit("--eval requires --restore_path: evaluating a random actor is "
                         "not an evaluation of anything.")
    return variant


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from examples.train_franka_service import main  # noqa: PLC0415

    main(build_variant(argparse.ArgumentParser(description=__doc__)))
