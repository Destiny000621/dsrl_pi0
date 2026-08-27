#!/bin/bash
# DSRL full-task baseline on YAM-abc — EARBUD INSERT (2026-08-25).
# Station deltas vs the retired vial script (run_yam.sh): FlexPoint grippers
# normalized [0,1] (open=1.0), earbud SFT/prompt, wandb project
# subrl-yam-earbud-insert. Hyperparameters unchanged (plan §5.1).
#
# Prerequisites:
#   1. GPU is FREE of training jobs: RLT stage-1 (train_rlt.py) and any SFT run
#      must be down — serve + SAC don't fit beside them. NEVER kill RustDesk.
#   2. pi0.5 serve (separate shell, limb/openpi venv). PLAIN serve — do NOT set
#      SUBRL_RLTOKEN for DSRL runs (it swaps the serve's SubRL embed hook to the
#      learned token; DSRL's get_prefix_rep is pinned to the OFFICIAL dsrl_pi0
#      last-prefix-slot feature either way, but keep the serves distinguishable):
#        cd ~/Desktop/research/limb/openpi && \
#        XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 \
#          uv run python scripts/serve_policy.py --port=8111 policy:checkpoint \
#          --policy.config=pi05_yam_abc_earbuds \
#          --policy.dir=/home/ssc/.cache/openpi/hf/pi05_yam_earbuds_teleop_15k
#   3. `wandb login` as destiny0621; limb checkout on branch YAM-abc.
#
# Mode 'keypress' (plan §4 Mode 0): 'c' starts an episode, 'q' aborts,
# '1'/'0' labels the outcome; operator re-stages between episodes (<=10 s,
# same convention as the SubRL earbud loop). Pure-VLA gate first: with
# --num_initial_traj_collect 999 nothing ever updates — collect >=10 episodes
# and check ~10% success (the earbud SFT solo baseline) before real RL.

proj_name=subrl-yam-earbud-insert
device_id=0

export EXP=./logs/dsrl_yam_abc_earbud
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# SAC shares the 5090 with the serve (0.6): keep the SAC slice small.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2

export remote_host=${remote_host:-127.0.0.1}
export remote_port=${remote_port:-8111}

python3 examples/launch_train_yam.py \
--mode keypress \
--prefix dsrl_yam_abc_earbud \
--launch_group_id dsrl-fulltask \
--wandb_project ${proj_name} \
--limb_config configs/yam_subtask_rl_earbud_insert.yaml \
--batch_size 256 \
--discount 0.999 \
--seed 0 \
--max_steps 500000 \
--eval_interval 2000 \
--log_interval 100 \
--checkpoint_interval 20000 \
--multi_grad_step 30 \
--resize_image 128 \
--action_magnitude 2.5 \
--query_freq 25 \
--action_horizon 50 \
--max_timesteps 3600 \
--hidden_dims 1024 \
--num_qs 2
