#!/bin/bash
# DSRL learner for the single-arm Franka FR3 (avantbot station).
#
# This starts ONLY the learner. The rollout loop is an avantbot agent
# (`franka_pi05_dsrl`) in its own process, and the frozen pi0.5 serve is a third.
# See DSRL_FRANKA_RUNBOOK.md for the full three-terminal procedure.
#
# Sized for a 100-EPISODE real-robot budget. Every default that differs from
# upstream is justified in launch_train_franka.py's docstring; the two that are
# MEASURED rather than chosen are noise_rows=50 and state_dim=2058.

set -euo pipefail

proj_name=DSRL_pi0_Franka
export EXP=${EXP:-./logs/$proj_name}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# The pi0.5 serve on the same GPU preallocates ~24.6 GB of a 32.6 GB card; this
# learner peaks at ~1.5 GB, so take it on demand instead of reserving a fraction.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

PY=${PY:-$HOME/venvs/dsrl/bin/python}

exec "$PY" -m examples.launch_train_franka \
  --num_episodes 100 \
  --port 9111 \
  --query_freq 50 \
  --noise_rows 50 \
  --state_dim 2058 \
  --num_initial_traj_collect 5 \
  --resize_image 128 \
  --num_cameras 2 \
  --batch_size 256 \
  --multi_grad_step 30 \
  --hidden_dims 1024 \
  --discount 0.9995 \
  --action_magnitude 2.0 \
  --num_qs 2 \
  --target_entropy 0.0 \
  --max_steps 200000 \
  --checkpoint_interval 10000 \
  --log_interval 100 \
  --seed 0 \
  --prefix dsrl_franka_cable \
  --wandb_project dsrl-franka-cable \
  "$@"
