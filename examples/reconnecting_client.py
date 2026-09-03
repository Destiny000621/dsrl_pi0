"""Serve client that survives :8111 restarts.

The station's serve process has been externally terminated repeatedly
(2026-08-26/-30, and mid-run on 2026-09-02, which wedged a 19 h training
process: openpi's WebsocketClientPolicy connects once in __init__ and never
reconnects, so a dead serve turns every later episode into an instant error).
This wrapper blocks until the serve is back (same semantics as the client's
own _wait_for_server) and retries the call once; a second failure raises into
collect_traj's episode-error path. While it blocks, the robot PD-holds its
last commanded target — safe, the arm simply freezes.
"""


class ReconnectingPolicyClient:
    def __init__(self, host, port):
        self._host, self._port = host, int(port)
        self._client = None
        self.connect()

    def connect(self):
        from openpi_client import websocket_client_policy as w

        if self._client is not None:
            try:
                self._client._ws.close()
            except Exception:
                pass
        print(f"[serve] connecting to {self._host}:{self._port} (blocks until the serve is up)...")
        self._client = w.WebsocketClientPolicy(host=self._host, port=self._port)
        print("[serve] connected")

    def get_server_metadata(self):
        return self._client.get_server_metadata()

    def _call(self, name, *args, **kw):
        try:
            return getattr(self._client, name)(*args, **kw)
        except Exception as e:
            print(f"[serve] {name} failed ({type(e).__name__}: {e}) — reconnecting "
                  "(robot HOLDS its last target meanwhile)")
            self.connect()
            return getattr(self._client, name)(*args, **kw)

    def infer(self, obs, noise=None):
        return self._call("infer", obs, noise=noise)

    def get_prefix_rep(self, obs):
        return self._call("get_prefix_rep", obs)
