# DSRL on the Franka FR3 — runbook (100-episode online RL)

DSRL (`nakamotoo/dsrl_pi0`, CoRL 2025) is the **full-task RL baseline**: pi0.5 stays
frozen and SAC steers its flow-matching latent. This runbook is for a
**100-episode** session on the double-cable task at the avantbot station.

`jaxrl2/` is untouched — algorithm frozen by scope. Everything here is embodiment
plumbing plus hyperparameters sized for 100 episodes rather than upstream's
500k-gradstep / 3M-step budgets.

Design rationale and the measurements behind the numbers:
`~/Desktop/SubRL/docs/dsrl_franka_port_plan.md`.

---

## 0. One-time setup

**a. openpi must serve the DSRL wire layer.** The stock serve *silently ignores*
a `noise` field — you would get a plain pi0.5 rollout that looks exactly like a
DSRL run. Use the branch:

```bash
cd ~/Desktop/openpi
git fetch origin && git checkout Franka_DSRL     # Destiny000621/openpi
```

Verify against a running serve (no robot needed; ~2 min):

```bash
.venv/bin/python scripts/dsrl_verify_noise.py --host 127.0.0.1 --port 8111
```
All five gates must pass. G3 prints the z_rl width — if it is not 2048, pass the
new `10 + width` to the learner as `--state_dim`.

**b. Learner venv** (jaxrl2's JAX cannot share a venv with the serve):

```bash
uv venv --python 3.11 ~/venvs/dsrl
uv pip install --python ~/venvs/dsrl/bin/python \
  "jax[cuda12]==0.5.3" "flax==0.10.2" "distrax==0.1.5" "optax==0.2.4" \
  "tensorflow-probability==0.25.0" "gym==0.26.2" "ml_collections==1.0.0" \
  "wandb==0.19.9" "opencv-python==4.11.0.86" matplotlib "numpy<2.2" \
  "orbax-checkpoint==0.11.1" einops
uv pip install --python ~/venvs/dsrl/bin/python \
  -e ~/Desktop/openpi/packages/openpi-client            # msgpack wire format
```

`requirements.txt` pins `jax==0.5.0`; we install **0.5.3** because the station's
GPU is a 5090 (Blackwell, sm_120) and 0.5.3 is the version already proven to serve
pi0.5 on it. The sim-only deps (LIBERO, robosuite, gym-aloha, dm-control,
tensorflow) are not installed — nothing on the real-robot path imports them.

**c. wandb** — optional but strongly recommended; without it there is no curve to
judge the run by. `jaxrl2/utils/wandb_config.py` (copy `wandb_config_example.py`)
or just `wandb login`.

---

## 1. Three processes

Everything runs on the **station's own 5090**. No second machine is involved — the
arm and the cameras are here, so the training loop could not live anywhere else
anyway; a remote GPU could only ever have hosted the offline gates.

```bash
# 1) frozen pi0.5 serve (terminal 1) — the default 0.75 preallocation is FINE,
#    the learner needs only ~1.5 GB and takes it with PREALLOCATE=false below.
cd ~/Desktop/openpi
uv run python scripts/serve_policy.py --port=8111 \
  policy:checkpoint --policy.config=pi05_franka_double_cable_100_r6_rawrot_wcrop \
  --policy.dir=/home/boyuan/.cache/openpi/hf/pi05_franka_double_cable_100_wcrop_10k

# 2) DSRL learner (terminal 2) — or just: bash examples/scripts/run_franka.sh
cd ~/Desktop/dsrl_pi0
XLA_PYTHON_CLIENT_PREALLOCATE=false ~/venvs/dsrl/bin/python \
  -m examples.launch_train_franka --num_episodes 100

# 3) robot loop (terminal 3)
cd ~/Desktop/Haply_Franka/vendor/avantbot
pixi shell -e droid-openpi
python -m avantbot.collect --config policy/franka_pi05_ee_fr3_dsrl
```

Sanity check before touching the arm:

```bash
~/venvs/dsrl/bin/python -m pytest examples/tests/test_franka_offline.py -q  # 5 gates, CPU, ~20 s
curl -s localhost:9111/healthz     # -> "base_policy": true, "total_traj": 0
```

The test suite is pinned to CPU so it is safe to run with the serve and a live
session already holding the GPU.

### Measured resource budget (one 5090, 32.6 GB)

| process | VRAM | note |
|---|---|---|
| pi0.5 serve | ~24.6 GB | default 0.75 preallocation; drop to `MEM_FRACTION=0.55` (~18 GB) only if something else needs room |
| DSRL learner | **1.53 GB peak**, 0.46 GB steady | measured at the full production config (batch 256, 128×128×6, hidden 1024×3, 1600-D action) |
| robot loop | ZED decode only | no model on the GPU |

The replay buffer is **host RAM, not VRAM**: 221 KB/transition, so 100 episodes is
**~0.9 GB** (3.0 GB of capacity reserved).

**Free 2.8 GB first if it is tight:** SubRL's SAM3 server
(`real_robot/services/serve_sam3_hf.py`) is not used by DSRL — DSRL's reward is
the operator keypress, there is no verifier in this baseline. Stop it for the
session.

### Gradient-step throughput (measured, under serve contention)

**123 steps/s — 8.1 ms/step.** So the update block between episodes is:

| UTD (`--multi_grad_step`) | steps/episode | wall clock |
|---|---|---|
| 10 | 400 | 3 s |
| 20 | 800 | 7 s |
| **30 (default)** | **1200** | **10 s** |

All three fit inside a scene reset, so UTD 30 costs nothing in operator time —
keep it. The **first** update block takes ~45 s: that is one-time JIT
compilation, not a problem.

---

## 2. The operator loop

One DSRL decision = one full 50-step chunk = **1.67 s** at 30 Hz. An episode is
capped at 2700 ticks (90 s ≈ 54 decisions).

| key | meaning |
|---|---|
| **1** | SUCCESS — posts to the learner (rewards `[-1…-1, 0]`), saves the recording **with** a SUCCESS marker, then homes |
| **0** | FAILURE — posts to the learner (rewards all `-1`), saves the recording **without** the marker, then homes |
| **h** | ABORT — dropped from the buffer *and* the recording discarded |
| **r** | resume → opens the next DSRL episode |

So each episode is: *watch → `1` or `0` → re-stage the scene → `r`*. The agent
drives the recorder at **both** edges: opening an episode starts the recording,
and the label stops/saves it, then the runner homes and pauses for the re-stage.

**Never press `space`/`s`/`d` yourself.** Both live incidents on 2026-09-05 were
the operator and the agent sharing the recorder's toggle vocabulary: first `s`
after `0` stamped a SUCCESS marker on a failed episode; then, in a session where
recording had never been started, the failure label's stop-toggle *started* one
(START_STOP is a toggle — its direction depends on state only the lifecycle can
guarantee). With the agent owning both edges, recorder state always equals
episode state and every toggle lands right. `1/0/h`, `[r]`, `[p]`, and `[q]` are
the entire interface. `agent.emit_recording_triggers: false` restores the fully
manual recorder if ever needed.

`[p]` then `[r]` mid-episode is a *pause*, not an episode boundary: the episode
(and its staged decisions) survive; one decision spans the pause with its chunk
dropped.

**Label failures with `0`, never `d`.** DSRL's reward is sparse −1 per decision, so
failure episodes are the majority of its training signal; discarding them throws
away most of the run. Use `h`/`d` only for episodes that must never be learned
from — an unrecoverable state, or any episode where you took over. (DSRL cannot
consume interventions: there is no action→noise inverse for a frozen flow policy,
so a takeover episode has no valid latent to credit.)

Episodes that hit the tick cap close themselves as FAILURE — upstream's timeout
convention (rewards all −1, masks all 1, so it bootstraps rather than being
treated as terminal). You do not have to sit through hopeless episodes.

Between episodes the learner runs its whole gradient block (~1200 steps, seconds
to tens of seconds). The agent posts off the control loop and waits for it before
opening the next episode, so it overlaps your scene reset.

---

## 3. Hyperparameters for a 100-episode budget

Defaults are already set in `launch_train_franka.py`. Our SFT is DSRL's **aloha**
lineage (horizon 50, absolute chunked actions), not pi0-DROID (horizon 10, joint
velocity), so shapes come from aloha and the episode/reward structure from the
real-robot script. Where they disagree on a *capacity* knob the real recipe wins —
aloha is a 3M-step sim run, we have 100 real episodes.

| knob | value | why |
|---|---|---|
| `query_freq` | **50** | = `action_horizon`. Both upstream recipes decide once per full chunk. 1.67 s/decision |
| `noise_rows` | **50** | MEASURED. 1 row → 40.7 mm chunk-vs-expert MAE, 50 → 3.3 mm (baseline 4.0). Upstream's `(1,32)` starts RL 6.7× off the SFT manifold |
| `state_dim` | **2058** | MEASURED (10 proprio + 2048 z_rl). Upstream's `8 + 2024` is wrong twice over |
| `num_cameras` | **2** | wrist + side. Both upstream recipes assume 3 → 9 channels; ours is 6 |
| `resize_image` | **128** | real lineage. aloha's 64 throws away the pixels a cm-scale insertion needs |
| `hidden_dims` | **1024** (×3) | real lineage. The MLP input carries a 2048-D embed and a 1600-D action; aloha's 128 would bottleneck it |
| `max_episode_steps` | **2700** (90 s) | the 100 demos average 38.4 s; ~2.3× headroom |
| `discount` | **0.9995** | → `0.9995^50 = 0.975`/decision → horizon ≈ 40 decisions ≈ one episode. aloha's 0.999 gives 0.951/decision = horizon 20, leaving early decisions nearly blind to the only reward there is. **The one knob changed by judgement, not measurement** — revert to 0.999 if the critic is unstable |
| `action_magnitude` | **2.0** | aloha lineage, and measured at only +17% chunk MAE vs N(0,1) (see §4). 2.5 (the DROID value) costs +66% |
| `multi_grad_step` | **30** | real lineage. ~40 decisions × 30 = ~1200 grad steps/episode, which fits inside the scene reset |
| `batch_size` | **256** | universal across every DSRL config |
| `num_qs` / reduction | **2 / min** | real lineage (REDQ-10/mean was sim-only) |
| `target_entropy` | **0.0** | both shipped real-robot configs |
| `num_initial_traj_collect` | **5** | upstream uses 1. At a 100-episode budget these warmup episodes are ALSO the frozen-pi0.5 baseline the run is judged against, and one episode is too thin to seed 5000 grad steps. 5% of the budget for a real baseline row |
| `max_steps` | **200,000** | a ceiling, not a target (100 × ≤54 × 30 = 162k). Also sizes the buffer to 13,333 transitions ≈ 3 GB, well above the ~4,000 this run collects, so it never hits upstream's 2×-permanent / 3×-transient resize |

Total expected: **~4,000 transitions** and **~125k gradient steps** across 100
episodes. That is a small dataset for SAC; the critic MLP's layer norm
(`use_layer_norm=True`) is what makes UTD 30 survivable. If Q values diverge, drop
`--multi_grad_step` to 20, then 10.

---

## 4. Reading the curve — one artefact to know about

The warmup episodes (1–5) run **pure `N(0,1)`** noise — upstream's `i == 0` branch
— so they measure the *frozen base policy through the whole DSRL stack*. From
episode 6 the actor takes over and the noise distribution becomes
`M·tanh(·)` with `M = action_magnitude`, which is a **different distribution**.

Measured chunk-vs-expert position MAE by magnitude (frozen policy, no learning):

| noise | std | pos MAE (mm) | vs N(0,1) |
|---|---|---|---|
| N(0,1) (warmup) | 1.00 | 3.07 | 1.00× |
| M = 1.0 | 0.63 | 2.77 | 0.90× |
| **M = 1.6** | **1.00** | **3.14** | **1.02×** |
| **M = 2.0** (default) | 1.26 | 3.60 | 1.17× |
| M = 2.5 | 1.57 | 5.09 | 1.66× |
| M = 3.0 | 1.88 | 9.25 | 3.01× |

So **expect a small step down in success rate at episode 6** that is not the
learner failing — it is the +17% distribution shift from the warmup's N(0,1) to
the actor's range. Judge learning from the trend after episode 6, against the
episode-1–5 baseline, not against episode 5 alone.

`M = 1.6` reproduces N(0,1)'s scale exactly (tanh(N(0,1)) has std 0.63, and
0.63 × 1.6 = 1.0). Use it if the base-policy episodes at 2.0 come in clearly below
the serve-only eval baseline; the cost is less room to steer beyond the prior.

Watch **`is_success`** and **`success_rate_10`**, not reward or Q — reward is −1
per decision by construction and tells you nothing.

---

## 5. Stopping, resuming, checkpoints

**Ctrl+C is safe.** SIGINT/SIGTERM saves actor/critic *and* the replay buffer
before exiting. Saves also happen every 10 episodes (`--save_every_episodes`) and
every 10,000 grad steps, all into `$EXP/dsrl_franka_cable_seed0/`:

```
checkpoint_<step>     actor / critic / target / temperature
replay_buffer.pkl     the transitions  (written to .tmp then renamed, so a
                      Ctrl+C mid-write cannot destroy the previous good one)
counters.json         total_traj, grad_steps, env_steps, per-episode successes
```

To continue in a later session, restore **both** — weights alone is not a
continuation:

```bash
D=$EXP/dsrl_franka_cable_seed0
~/venvs/dsrl/bin/python -m examples.launch_train_franka \
  --restore_path $D --restore_buffer $D/replay_buffer.pkl \
  --num_initial_traj_collect 0
```

`--num_initial_traj_collect 0` skips the base-policy warmup on a resume — you
already have that baseline row, and at 100 episodes you cannot afford to buy it
twice.

Upstream never persists the buffer on the real-robot path. That is survivable at
its 500k-gradstep budget and not at 100 real episodes, where the transitions are
robot time and cannot be regenerated. (The restore here uses plain `pickle.load`
rather than jaxrl2's `ReplayBuffer.restore`, which reads a `pickle.dump` file
with `np.load(..., allow_pickle=True)[0]` and is marked "todo test this".)

## 6. Evaluating the trained policy

Same three processes; only the learner's flags and the session file change.
Training and eval never share a config.

```bash
# terminal 2 — learner in EVAL mode (no learning, no saves; actor drives all)
cd ~/Desktop/dsrl_pi0
D=$(pwd)/logs/dsrl_franka/dsrl_franka_cable_seed0
XLA_PYTHON_CLIENT_PREALLOCATE=false ~/venvs/dsrl/bin/python \
  -m examples.launch_train_franka --eval 1 --restore_path $D \
  --prefix dsrl_franka_eval

# terminal 3 — eval session (recordings go to data_log_dsrl_eval)
cd ~/Desktop/Haply_Franka/vendor/avantbot && pixi shell -e droid-openpi
python -m avantbot.collect --config policy/franka_pi05_ee_fr3_dsrl_eval
```

Operator flow is unchanged — `1`/`0` label each episode, `r` opens the next —
and the learner prints the running tally after every episode:

```
EVAL episode 12: SUCCESS | running success 7/12 = 58.3%
```

That tally is the evaluation. Notes:

- **Sampled, not deterministic, by default.** Upstream DSRL never evaluates with
  the actor mean — rollouts and evals both `sample_actions` — so sampling is the
  convention-faithful number to report against other DSRL results.
  `--eval_deterministic 1` reports the policy's *mode* instead, which is a
  different question.
- There is **no N(0,1) warmup in eval** and nothing enters the buffer; `/healthz`
  shows `"mode": "eval"`.
- To eval a mid-training snapshot, point `--restore_path` at the run dir — flax
  picks the newest `checkpoint_<step>` — while the training learner stays
  stopped (one GPU, one serve, and the two learners must not share :9111).
- The frozen-pi0.5 **baseline row needs no eval mode**: it is episodes 1–5 of
  training (pure N(0,1)), or a plain `franka_pi05_ee_fr3_wcrop` eval session.

## 7. Failure modes worth recognising

| symptom | cause |
|---|---|
| rollout looks like plain pi0.5, learner shows `total_traj: 0` | serve is not on `Franka_DSRL`, or the session YAML lost `noise_provider` wiring. Run the gates |
| agent refuses to start, complains about `chunk_size_threshold` | the YAML was copied from the eval session. DSRL needs `0.0` + `blend_mode: latest_only`, one latent per chunk |
| `state is N-D but the learner was built for 2058` | z_rl width changed (different serve flags). Re-measure with G3, relaunch with `--state_dim` |
| episode logged as ABORTED without you pressing `h` | a learner call failed mid-episode; the agent fell back to `N(0,1)` and refuses to train on latents the actor did not choose. Check terminal 2 |
| OOM on the learner | it needs only ~1.5 GB, so something else filled the card. Stop SubRL's SAM3 server (2.8 GB), or start the serve with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.55` |
| "Waiting for the learner's update block" for ~2 min after episode 5 | the one-time 5000-step warmup + JIT (measured live: 101 s); later blocks are 2–10 s. The agent logs progress every 15 s |
| `Checkpoint path should be absolute` warnings, no `checkpoint_*` on disk | you are on a build older than 2026-09-05 — the launcher now abspaths `EXP`. Weights from such a run are unrecoverable, but the buffer is not: resume with `--restore_buffer` alone and the counter resets so the warmup block **retrains actor/critic from the buffer** (~2 min) |
| resumed with `--restore_buffer` but not `--restore_path`, and it retrains ~2 min at startup | intentional: grad_steps describes the *weights*; without them the counter resets and the buffer retrains a fresh actor **before the robot connects**, so zero robot episodes are spent on it and none run N(0,1) |
| a real `checkpoint<step>` vanished right after being saved | stale `*.orbax-checkpoint-tmp-*` dirs from a crashed save outrank it in flax's name-parsed retention; the learner now purges them at startup |
| startup says "base-policy phase: next N episodes" after a restore | that banner now prints the actual remaining count from the restored state, not the flag; "actor is LIVE from the first episode" means zero N(0,1) episodes will be collected |
| `Found no checkpoint files ... with prefix checkpoint_` | you are on a build older than this fix: jaxrl2 saves prefix `checkpoint` but flax restores prefix `checkpoint_`, so `--restore_path` silently no-opped and the session (or an eval!) ran on fresh weights while logging "restored from ...". The learner now resolves the newest `checkpoint<step>` file itself, adopts its step as grad_steps, and hard-fails if none exists |
| "Episode POST outlived its timeout" | the learner is still processing — the episode **does** land in the buffer (verified live: a "lost" SUCCESS was there all along); only the acknowledgement died. If it recurs, the learner is far too slow — check its terminal and the GPU |
| an episode closes with 0 decisions the instant [r] opens it | fixed: keys pressed during a pause/wait were consumed as labels on resume; the agent now discards stale keypresses at reset |
