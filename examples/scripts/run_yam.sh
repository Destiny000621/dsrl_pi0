#!/bin/bash
# DSRL on YAM — full-task baseline (SubRL-VLA/docs/dsrl_yam_port_plan.md §5.1).
#
# Prerequisites (plan §6):
#   1. limb/openpi serve on this box (separate shell, limb/openpi venv):
#        SUBRL_RETURN_EMBED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 \
#          uv run python scripts/serve_policy.py --port=8111 policy:checkpoint \
#          --policy.config=pi05_yam_vial_4_30fps_aug \
#          --policy.dir=$HOME/.cache/openpi/hf/yam-vial-aug-pi05-v1-10k
#   2. `wandb login` as destiny0621 in this venv (jaxrl2 falls back to the
#      ambient login when jaxrl2/utils/wandb_config.py is absent).
#   3. limb installed into this venv (pip install -e /home/ssc/Desktop/research/limb).
#
# Mode 'keypress' (bring-up, plan §4 Mode 0): operator stages the scene and
# presses 'c' to start each episode, labels the outcome 1/0, 'q' aborts an
# episode. The i==0 block (first episode + 5000 grad steps) doubles as the
# pure-N(0,1) base-policy baseline — record >=10 such episodes (plan §6.6)
# before letting updates begin, by restarting with num_initial_traj_collect.

proj_name=subrl-yam-grasp
device_id=0

export EXP=./logs/dsrl_yam
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# SAC shares the 5090 with the pi0.5 serve process (plan §5.2): serve ~0.65, SAC ~0.2.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2

# pi0.5 serve endpoint (same box by default).
export remote_host=${remote_host:-127.0.0.1}
export remote_port=${remote_port:-8111}

python3 examples/launch_train_yam.py \
--mode keypress \
--prefix dsrl_yam_fulltask \
--launch_group_id dsrl-fulltask \
--wandb_project ${proj_name} \
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
--max_timesteps 1200 \
--hidden_dims 1024 \
--num_qs 2
