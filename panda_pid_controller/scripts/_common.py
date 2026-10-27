"""Shared helpers for the demo scripts."""
import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--viewer", action="store_true", help="open the interactive MuJoCo viewer")
    return ap


@contextmanager
def viewer_callback(env, enabled: bool):
    """Yield a per-step callback that keeps the MuJoCo viewer in sync (or ``None`` if disabled)."""
    if not enabled:
        yield None
        return
    import mujoco.viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        def cb(_env):
            viewer.sync()
            time.sleep(_env.dt)
            if not viewer.is_running():
                raise KeyboardInterrupt
        yield cb
