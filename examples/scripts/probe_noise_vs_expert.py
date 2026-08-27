"""Does DSRL's tiled single-row noise change pi0.5-SFT behavior vs i.i.d. sampling?
Probes expert-demo frames (earbuds_30fps_v21) against the live serve: for each frame,
6 chunks with i.i.d. N(0,1) (50,32) noise [= SFT sampling] and 6 with one N(0,1) row
tiled across the horizon [= DSRL base-policy phase]. Metrics on the EXECUTED part
(first 25 steps), joint dims only (12) and gripper dims (2) separately:
  err_*   = mean |chunk - expert| ;  spread_* = mean pairwise |a - b| within a group ;
  cross   = mean |iid_i - tiled_j|.
"""
import cv2, numpy as np, pandas as pd, sys
from openpi_client import image_tools, websocket_client_policy as W

D = "/home/ssc/Desktop/research/limb/datasets/earbuds_30fps_v21"
PROMPT = "insert the wireless bluetooth earbuds into the charging case"
CAMS = {"cam_high": "observation.images.head_camera", "cam_left_wrist": "observation.images.left_wrist_camera",
        "cam_right_wrist": "observation.images.right_wrist_camera"}
Q, H, NOISE_DIM, N = 25, 50, 32, 6
J = np.r_[0:6, 7:13]; G = np.r_[6, 13]

def frame(ep, key, t):
    cap = cv2.VideoCapture(f"{D}/videos/chunk-000/{key}/episode_{ep:06d}.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, t); ok, im = cap.read(); cap.release()
    assert ok, (ep, key, t)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def wire_obs(ep, t, state):
    return {"state": state.astype(np.float32),
            "images": {k: np.transpose(image_tools.resize_with_pad(frame(ep, v, t), 224, 224), (2, 0, 1)) for k, v in CAMS.items()},
            "prompt": PROMPT}

def mae(a, b, dims): return float(np.abs(a[:Q, dims] - b[:Q, dims]).mean())
def pair_spread(S, dims): return float(np.mean([mae(S[i], S[j], dims) for i in range(len(S)) for j in range(i + 1, len(S))]))

cli = W.WebsocketClientPolicy(host="127.0.0.1", port=8111)
rng = np.random.default_rng(0)
rows = []
for ep in (0, 7, 19):
    df = pd.read_parquet(f"{D}/data/chunk-000/episode_{ep:06d}.parquet")
    st = np.stack(df["observation.state"].to_numpy()); ac = np.stack(df["action"].to_numpy())
    T = len(df)
    for frac in (0.2, 0.5, 0.8):
        t = int(frac * (T - H - 1)); obs = wire_obs(ep, t, st[t]); expert = ac[t:t + H]
        iid = [np.asarray(cli.infer(obs, noise=rng.standard_normal((1, H, NOISE_DIM)).astype(np.float32))["actions"]) for _ in range(N)]
        tiled = [np.asarray(cli.infer(obs, noise=np.repeat(rng.standard_normal((1, NOISE_DIM)).astype(np.float32), H, 0)[None])["actions"]) for _ in range(N)]
        r = dict(ep=ep, t=t,
                 errJ_iid=np.mean([mae(c, expert, J) for c in iid]), errJ_tiled=np.mean([mae(c, expert, J) for c in tiled]),
                 sprJ_iid=pair_spread(iid, J), sprJ_tiled=pair_spread(tiled, J),
                 crossJ=np.mean([mae(a, b, J) for a in iid for b in tiled]),
                 errG_iid=np.mean([mae(c, expert, G) for c in iid]), errG_tiled=np.mean([mae(c, expert, G) for c in tiled]),
                 sprG_iid=pair_spread(iid, G), sprG_tiled=pair_spread(tiled, G))
        rows.append(r)
        print(f"ep{ep:02d} t={t:4d} | joints rad: err iid {r['errJ_iid']:.4f} tiled {r['errJ_tiled']:.4f} | spread iid {r['sprJ_iid']:.4f} tiled {r['sprJ_tiled']:.4f} cross {r['crossJ']:.4f} | grip: err iid {r['errG_iid']:.3f} tiled {r['errG_tiled']:.3f}", flush=True)
m = pd.DataFrame(rows).drop(columns=["ep", "t"]).mean()
print("\nMEAN over 9 probes (first 25 steps):")
print(f"  joints  err-vs-expert : iid {m.errJ_iid:.4f}  tiled {m.errJ_tiled:.4f} rad   (ratio {m.errJ_tiled/m.errJ_iid:.2f})")
print(f"  joints  sample spread : iid {m.sprJ_iid:.4f}  tiled {m.sprJ_tiled:.4f}  cross {m.crossJ:.4f} rad")
print(f"  gripper err-vs-expert : iid {m.errG_iid:.3f}  tiled {m.errG_tiled:.3f}   spread iid {m.sprG_iid:.3f} tiled {m.sprG_tiled:.3f}")
