"""Replay-buffer persistence for the YAM DSRL loop.

Upstream jaxrl2 keeps the buffer only in RAM, so a process death loses every
robot transition (2026-09-02: a wedged run held 29 episodes / 1121 decisions
that had to be salvaged from the live process). save_buffer runs at the same
cadence as SAC checkpoints; load_buffer accepts both these saves and the
gdb-salvaged dump format (a [payload] list with identical keys).
"""
import os
import pickle


def _trim(node, n):
    if isinstance(node, dict):
        return {k: _trim(v, n) for k, v in node.items()}
    return node[:n].copy()


def save_buffer(buffer, path):
    """Atomic pickle of the buffer's rows [0:size] + counters (~324 KB/row)."""
    payload = {
        "data": _trim(buffer.data, buffer.size),
        "size": int(buffer.size),
        "_traj_counter": int(buffer._traj_counter),
        "_start": int(buffer._start),
        "traj_bounds": dict(buffer.traj_bounds),
    }
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    os.replace(tmp, path)


def _copy_into(dst, src):
    for k, v in src.items():
        if isinstance(v, dict):
            _copy_into(dst[k], v)
        else:
            dst[k][: v.shape[0]] = v


def load_buffer(buffer, path):
    """Load a save (or salvage dump) into a freshly constructed buffer."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    if isinstance(d, list):  # gdb salvage wraps the payload in a list
        d = d[0]
    n = int(d["size"])
    if n > buffer.capacity:
        raise ValueError(f"dump has {n} rows > buffer capacity {buffer.capacity}")
    _copy_into(buffer.data, d["data"])
    buffer.size = n
    buffer._traj_counter = int(d["_traj_counter"])
    buffer._start = int(d["_start"])
    buffer.traj_bounds = {int(k): tuple(v) for k, v in dict(d["traj_bounds"]).items()}
    return n, buffer._traj_counter
