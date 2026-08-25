# Verifying the DSRL baseline on YAM — bring-up runbook

Step-by-step verification of the DSRL full-task baseline, from a fresh venv to
the first RL run. Each stage has a **pass criterion** — do not advance until it
holds. Companion to the port plan
(`SubRL-VLA/docs/dsrl_yam_port_plan.md`, §6 bring-up); code lives on:

- `Destiny000621/dsrl_pi0`, branch **`yam-fulltask`** (this repo — SAC loop + YamEnv)
- `Destiny000621/openpi`, branch **`dsrl_yam`** (serving: noise envelope + `get_prefix_rep`)

Topology (one machine, the YAM box): pi0.5 SFT serves frozen on `:8111`
(~0.6 of the 5090's VRAM), the DSRL SAC + robot env run in this repo's venv
(~0.2), cameras/arms attach over Portal/CAN.

---

## ⚠ Station update 2026-08-25 — YAM-abc (earbud insert)

The stages below were first verified on the retired vial station. The live
station is **YAM-abc** (limb branch `YAM-abc`; two-arm ex-leaders, FlexPoint
grippers, all-D405 cams — never mix the stations' values). Code defaults now
target YAM-abc; where a value below conflicts with this table, this table wins.

| Item | YAM-abc value (vs vial-era) |
|---|---|
| Task / SFT | **earbud insert**, `pi05_yam_abc_earbuds`, ckpt `~/.cache/openpi/hf/pi05_yam_earbuds_teleop_15k` (PINNED; was vial `pi05_yam_vial_4_30fps_aug`) |
| Prompt (verbatim, no trailing period) | `insert the wireless bluetooth earbuds into the charging case` |
| Grippers | FlexPoint, **normalized [0,1], open=1.0** (was [0,2.4], open=2.2) — obs AND commands; stage 2 asserts obs gripper dims land in [0,1] |
| Cameras | same names (head/left_wrist/right_wrist → cam_high/left/right), all D405, NEW serials 409122274017 / 427622273576 / 427622271888 (from the limb YAML — nothing to change in DSRL) |
| limb config | `configs/yam_subtask_rl_earbud_insert.yaml` (robots/sensors sections only) |
| Run script | `examples/scripts/run_yam_abc_earbud.sh` (`run_yam.sh` = retired vial station) |
| wandb | project **`subrl-yam-earbud-insert`**, group `dsrl-fulltask` (cross-baseline row vs RLT `rlt_fulltask_v1` and SubRL `earbud_insert_v*`) |
| Stage-3 gate | **~10%** pure-VLA success (earbud SFT solo baseline; was 6/23 ≈ 26% for vial) |
| Episode | 1200 steps / 40 s @30 Hz (matches the SubRL earbud RL budget); operator re-stage ≤10 s |

Serve on YAM-abc (plain — do **not** set `SUBRL_RLTOKEN` for DSRL sessions;
`get_prefix_rep` is pinned to the plain mean-pool either way, but keep the
serve variants distinguishable):

```bash
cd ~/Desktop/research/limb/openpi     # yam-vial-30fps-v1 + uncommitted YAM-abc configs
XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 \
uv run python scripts/serve_policy.py --port=8111 policy:checkpoint \
  --policy.config=pi05_yam_abc_earbuds \
  --policy.dir=/home/ssc/.cache/openpi/hf/pi05_yam_earbuds_teleop_15k
```

GPU cohabitation on this box: check `nvidia-smi` FIRST — RLT stage-1 training
(`train_rlt.py`, ~22 GB) and SFT runs are mutually exclusive with the serve;
SAM3 (:8114) / DINOv2 (:8118) may be resident for SubRL work; **never kill
RustDesk**. Ports 8112/9107–9109 belong to the RLT baseline.

Venv addendum for stages 2+ (robot path; verified 2026-08-25 — the July venv
lacked these). Order matters:

```bash
conda activate dsrl_yam
pip install "robocam @ git+https://github.com/TToTMooN/robocam.git@7284ce3a7fb762ac9497b95beea8782095c9a7a2"
pip install python-can ruckig      # ruckig wheel BEFORE i2rt (i2rt's pinned 0.15.3 has no wheel; source build fails)
pip install -e ~/Desktop/research/limb/dependencies/i2rt --no-deps   # its exact pins (numpy/mujoco/portal) would wreck this env
pip install "qpsolvers[quadprog]" threadpoolctl "click<8.2.0"
pip install mujoco==3.3.5          # yam.xml needs mujoco>=3 (2.3.7 fails at 'joint')
# ^ knowingly breaks dm-control's mujoco pin: this venv is YAM-REAL-ONLY —
#   the upstream ALOHA/LIBERO sim scripts will not run in it. (i2rt's tyro
#   pin warning is benign; limb uses tyro 1.x.)
# verify:
python -c "from limb.utils import launch_utils; from i2rt.robots.motor_chain_robot import MotorChainRobot; import mujoco; mujoco.MjModel.from_xml_path('$HOME/Desktop/research/limb/dependencies/i2rt/i2rt/robot_models/yam/yam.xml'); print('robot path OK')"
```

---

## Stage 0 — one-time setup

### 0.1 dsrl venv

```bash
# -c conda-forge --override-channels: anaconda's default channels are gated
# behind a Terms-of-Service acceptance on this machine; conda-forge is not.
conda create -n dsrl_yam -c conda-forge --override-channels python=3.11.11 -y
conda activate dsrl_yam
cd ~/Desktop/research/dsrl_pi0            # branch yam-fulltask
pip install -e . -r requirements.txt      # moviepy/opencv/dm-env/msgpack are in the pins
pip install "jax[cuda12]==0.5.0"

# openpi-client MUST come from limb/openpi branch dsrl_yam — it has the noise
# kwarg + dict-returning get_prefix_rep. Do NOT install the nakamotoo submodule's
# client (tuple-returning get_prefix_rep; incompatible with train_utils_yam).
pip install -e ~/Desktop/research/limb/openpi/packages/openpi-client

# limb as a library: --no-deps, then the runtime deps our path actually imports —
# a full `pip install -e limb` would pull pyroki, whose old jax pin downgrades
# the env. (The dangling `yourdfpy` warning is limb's viser path; never imported.)
pip install -e ~/Desktop/research/limb --no-deps
pip install loguru portal omegaconf tyro pyrealsense2

wandb login    # as destiny0621 (needed from stage 3 on, not for stages 1-2)
```

Verified 2026-07-26 on the YAM box: jax 0.5.0+cuda12 works on the 5090
(sm_120 JIT-compiles via nvjitlink — the plan §0.4 fallback was not needed).
The cuFFT/cuDNN "factory already registered" lines at import are TF+JAX
coexistence noise, harmless. If a future wheel bump crashes with a PTX/sm_120
error: `pip install "jax[cuda12]==0.5.3"`, or `JAX_PLATFORMS=cpu` as a last
resort (the 128 px CNN is small).

### 0.2 pi0.5 serve (separate shell, limb/openpi venv)

```bash
cd ~/Desktop/research/limb/openpi         # branch dsrl_yam
XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 \
uv run python scripts/serve_policy.py --port=8111 policy:checkpoint \
  --policy.config=pi05_yam_vial_4_30fps_aug \
  --policy.dir=$HOME/.cache/openpi/hf/yam-vial-aug-pi05-v1-10k

curl http://localhost:8111/healthz        # -> "OK"
```

Notes: `--port` must precede `policy:checkpoint`; first request takes ~30 s
(JIT), then ~160 ms. `SUBRL_RETURN_EMBED=1` is NOT needed — DSRL uses the
`get_prefix_rep` method, which works regardless of that flag.

---

## Stage 1 — wire layer (NO robot)

```bash
cd ~/Desktop/research/dsrl_pi0
python examples/scripts/verify_wire.py
```

Checks: plain infer still works (backward compat with the SubRL stack, so the
same server can keep serving both), same-noise → identical (50, 14) chunks,
no-noise → differing chunks, the DSRL tiled-row noise structure, and
`get_prefix_rep` → float32 `(1, emb)` (prints the SAC `state_dim`,
expected 14 + 2048 = 2062).

**Pass:** script prints `ALL WIRE CHECKS PASSED`.
Failure modes: "same noise gave different chunks" → the serve is not running
the `dsrl_yam` branch; envelope/`get_prefix_rep` errors → stale openpi-client
install (re-run the 0.1 `pip install -e .../openpi-client`).

Optional deeper check: replay a recorded obs from
`limb/recordings/subrl_eval_full_task_*` through plain infer and diff against
the pre-patch serve output for the same obs — proves the patch changed nothing
on the plain path.

## Stage 2 — YamEnv dry-run (ROBOT live, no RL)

Arms powered, CAN up, three RealSense cams connected. **Stand clear.**

```bash
python examples/scripts/verify_yam_env.py
```

Checks, with a deliberately tight 0.05 rad/tick clamp so nothing can move fast:
obs pipeline shapes (14-D qpos, CHW-224 wire images, (1,128,128,9,1) SAC
pixels), `reset()` safe-move + measured-pose re-anchor, the zeros-action hold
sentinel (no motion for 1 s), an adversarial ±π action (must produce at most a
0.05 rad twitch and clipped grippers — this is the safety chain that prevents
robot damage), and EE-pose FK injection. Ends with a soft-release shutdown.

**Pass:** `ALL YAM-ENV CHECKS PASSED`, and during the run you visually
confirmed: reset moves are slow, zeros action does not move the arms, the ±π
action barely twitches. If you ever see
`safe-move may have failed` afterwards, check the arm before continuing —
the clamp re-anchors to the measured pose, so it stays safe, but the scene
start state is wrong.

## Stage 3 — base-policy episodes through the full DSRL loop (the key gate)

Run the real training entry point but **hold updates off** so every episode
uses N(0,1) noise = the frozen pi0.5 base policy through the complete
obs→prefix_rep→noise→chunk→safety-chain stack:

```bash
# same shell setup as run_yam.sh, but updates gated off:
export EXP=./logs/dsrl_yam CUDA_VISIBLE_DEVICES=0 \
       XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
python examples/launch_train_yam.py --mode keypress \
  --prefix dsrl_yam_basepolicy --wandb_project subrl-yam-grasp \
  --num_initial_traj_collect 999 \
  --max_timesteps 1200 --query_freq 25 --action_horizon 50
```

Operator loop per episode: stage the vial scene → `c` starts → (up to 40 s at
30 Hz; `q` aborts) → label `1`/`0` → arms auto-return home → re-stage.
Episode videos land in `$EXP/<run>/video_high_<n>.mp4`.

Collect **≥ 10 episodes**, then Ctrl-C (`num_initial_traj_collect=999` means no
gradient step ever runs — this run is purely the baseline).

**Pass:**
- success rate over ≥10 episodes ≈ the serve-only eval baseline
  (6/23 ≈ 26% from `subrl_eval_full_task_*`; anything in ~15–40% is consistent
  at this sample size). **Much lower means the port broke the policy — stop
  and debug before any RL** (first suspects: image CHW order, qpos layout,
  prompt string, delta-clamp too tight — watch for constant clamp warnings).
- videos look like normal pi0.5 behavior (smooth approach, no jerks, no
  gripper flutter);
- wandb run (project `subrl-yam-grasp`, group `dsrl-fulltask`) shows
  `is_success`, `episode_reward`, `episode_steps`, and `success_rate_10`
  after episode 10.

These ≥10 episodes are the "BC init" row of the baseline table — record the
number in the run notes.

## Stage 4 — first RL run

```bash
bash examples/scripts/run_yam.sh
```

This is the plan §5.1 config: `query_freq 25`, `discount 0.999`,
`action_magnitude 2.5`, hidden 1024×3, `num_qs 2`, batch 256, UTD 30,
1200-step episodes, 500k grad-step ceiling. The first episode + 5000-grad-step
block is the built-in warmup; after that, expect ~1440 grad steps (≈1–2 min on
the 5090) between episodes.

**Watch in the first hour:**
- `training/critic_loss`, `training/actor_loss` finite, no NaN;
  `training/temperature` moving (target_entropy 0.0);
- replay-buffer RAM ≈ 311 KB/transition (~10 GB at the full 33k capacity —
  fine on this box, but do not raise `max_steps` without re-doing that math);
- robot: exploration is wider than base policy (that is the point of
  `action_magnitude 2.5`) but must stay smooth — if motion looks erratic,
  restart with `--action_magnitude 2.0` (the ALOHA-sim precedent);
- checkpoints appear at `$EXP/<run>/` every 20k grad steps;
- `perform_eval skipped: ...` lines at eval intervals are expected (upstream's
  Q-visualization is 3-channel-only), harmless.

**Pass (learning signal):** `success_rate_10` trending above the stage-3
baseline within the first ~50–100 episodes (DSRL precedent: VINE's table shows
17/20 on plug insertion after ~50 min of robot time; the EXPO-FT paper's DSRL
reached 19/30 average). No hard threshold — the curve direction is the check.

---

## Quick troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Unrecognized options: --port` at serve | `--port` must come before `policy:checkpoint` |
| assertion `max > 1` / scrambled images server-side | images sent HWC or float — must be CHW uint8 (stage 1/2 verify this) |
| `same noise gave different chunks` | serve not on `dsrl_yam` branch |
| `TypeError: infer() got ... 'noise'` | stale openpi-client — reinstall from limb/openpi |
| msgpack error on prefix_rep | server not float32-casting → wrong branch |
| SAC init crash mentioning `sm_120`/PTX | jax 0.5.0 lacks Blackwell kernels → jax 0.5.3 or CPU |
| serve or SAC OOM | re-check `XLA_PYTHON_CLIENT_MEM_FRACTION` split 0.65/0.2; Mode-0 needs no SAM3 |
| constant `[YamEnv] ... clamp` warnings + sluggish tracking | delta clamp too tight for real motion — raise `--joint_delta_limit` toward 0.15–0.2 |
| `safe-move may have failed` after reset | arm did not reach home; check hardware — the clamp stays anchored to the measured pose (safe), but fix the scene before `c` |
| i2rt `RuntimeError` kills a robot subprocess | a raw joint-limit violation got through — this should be impossible via the clamp; capture logs and stop |
| run dies between episodes with buffer `randint` error | should be fixed (empty-trajectory guard); if seen, report — an episode produced 0 decisions |

## Out of scope here (next milestones)

- **Autonomous mode (plan §5.3b/§5.4):** author + calibrate the full-task
  insertion verifier against the 23 labeled `subrl_eval_full_task_*` episodes,
  then the hybrid reset (EAP place-back on failure / operator re-stage after
  success). `--mode autonomous` fails fast until then — by design.
- **Pedal safety mode** (`--mode pedal_safety`): phase 2; takeover transitions
  can never become DSRL training data (no action→noise inverse).
