"""Stage 1 — wire-layer verification, NO ROBOT (docs/yam_verification.md).

Checks against a live pi0.5 serve (:8111 from limb/openpi branch dsrl_yam):
  1. plain infer (no noise) still works           -> backward compat, (50, 14) chunk
  2. same obs + same explicit (1,50,32) noise x2  -> identical action chunks
  3. same obs + noise=None x2                     -> different chunks (server RNG)
  4. tiled single-row noise (the DSRL structure)  -> accepted, (50, 14) chunk
  5. get_prefix_rep                               -> float32 (1, emb); prints emb width

Run inside the dsrl venv:
    python examples/scripts/verify_wire.py [--host 127.0.0.1] [--port 8111]
"""
import argparse

import numpy as np
from openpi_client import websocket_client_policy

# Default = YAM-abc earbud SFT default_prompt, verbatim (no trailing period).
# Pass --prompt to match whatever SFT the serve is running.
DEFAULT_PROMPT = "insert the wireless bluetooth earbuds into the charging case"


def synthetic_obs(prompt):
    return {
        "state": np.zeros(14, dtype=np.float32),
        "images": {
            name: np.zeros((3, 224, 224), dtype=np.uint8)
            for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        },
        "prompt": prompt,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8111)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = p.parse_args()

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print("server metadata:", client.get_server_metadata())
    obs = synthetic_obs(args.prompt)

    # 1. plain path (what the SubRL stack sends) — proves envelope backward compat
    a_plain = np.asarray(client.infer(obs)["actions"])
    assert a_plain.shape == (50, 14), f"plain infer chunk shape {a_plain.shape}, expected (50, 14)"
    print(f"1. plain infer OK: chunk {a_plain.shape}")

    # 2. explicit-noise determinism (first call may include JIT compile, ~30 s)
    noise = np.random.default_rng(0).standard_normal((1, 50, 32)).astype(np.float32)
    a1 = np.asarray(client.infer(obs, noise=noise)["actions"])
    a2 = np.asarray(client.infer(obs, noise=noise)["actions"])
    dmax = float(np.abs(a1 - a2).max())
    assert dmax < 1e-5, f"same noise gave different chunks (max diff {dmax}) — noise is NOT steering"
    assert a1.shape == (50, 14)
    print(f"2. noise determinism OK: max diff {dmax:.2e}")

    # 3. server RNG still varies without noise
    b1 = np.asarray(client.infer(obs)["actions"])
    b2 = np.asarray(client.infer(obs)["actions"])
    assert np.abs(b1 - b2).max() > 1e-4, "noise=None twice gave identical chunks — server RNG broken"
    print(f"3. RNG variability OK: max diff {np.abs(b1 - b2).max():.3f}")

    # 4. the exact noise structure the DSRL loop sends: one row tiled to 50
    row = np.random.default_rng(1).standard_normal((1, 32)).astype(np.float32)
    tiled = np.repeat(row, 50, axis=0)[None]
    a_tiled = np.asarray(client.infer(obs, noise=tiled)["actions"])
    assert a_tiled.shape == (50, 14)
    print(f"4. tiled-row noise OK: chunk {a_tiled.shape}")

    # 5. prefix rep for the SAC state
    rep = np.asarray(client.get_prefix_rep(obs)["prefix_rep"])
    assert rep.dtype == np.float32 and rep.ndim == 2 and rep.shape[0] == 1, rep.shape
    print(f"5. get_prefix_rep OK: shape {rep.shape} -> SAC state_dim will be {14 + rep.size}")

    print("\nALL WIRE CHECKS PASSED")


if __name__ == "__main__":
    main()
