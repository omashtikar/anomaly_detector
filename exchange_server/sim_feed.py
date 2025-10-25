# sim_feed.py
# Real-time jump-diffusion price + volume generator with optional WebSocket streaming.
import math
import time
import json
import signal
import argparse
from dataclasses import dataclass, asdict
from typing import Iterator, Dict, Any, Optional

import numpy as np

# ---------- Model configuration ----------

@dataclass
class JDParams:
    # Price process (Merton jump-diffusion)
    s0: float = 100.0           # initial price
    mu: float = 0.08            # drift per day (log)
    sigma: float = 0.25         # diffusive volatility per day (log)
    jump_lambda: float = 0.08   # expected jumps per day
    jump_mu: float = 0.0        # mean jump size in log returns
    jump_sigma: float = 0.03    # std of jump size (log)

    # Volume process (log-volume = seasonality + AR(1) + |ret|-sensitivity + jump bump + noise)
    v0: float = 150_000.0
    phi: float = 0.85           # AR(1) persistence (0..1)
    beta: float = 22.0          # sensitivity to |r_t|
    gamma: float = 0.7          # extra bump if jump occurs
    sigma_v: float = 0.35       # noise std in log-volume

    # Clock/time
    dt: float = 1.0             # seconds between ticks
    trading_seconds: float = 6.5 * 3600  # one "day" length
    seed: Optional[int] = 42    # RNG seed (None => nondeterministic)


def u_shape_profile(t_norm: float, a: float = 0.6, b: float = 0.6) -> float:
    """
    Intraday U-shaped seasonality profile in [~0, ~2].
    t_norm in [0,1]; larger at open and close.
    """
    # Normalize so midday ~ 0.5 and edges ~ 1.0
    base = (t_norm ** a) * ((1.0 - t_norm) ** b)
    return 0.5 + 0.5 * (base / 0.25)  # 0.25 is approx peak of Beta(a+1, b+1) for a=b~0.6


# ---------- Streaming generator ----------

def jd_price_volume_stream(params: JDParams) -> Iterator[Dict[str, Any]]:
    """
    Infinite generator that yields {"ts", "price", "ret", "jump", "volume"} every dt seconds.
    Scales mu/sigma/jump_lambda from "per day" to the chosen dt.
    """
    rng = np.random.default_rng(params.seed)
    S = float(params.s0)
    logV = math.log(max(params.v0, 1.0))

    # Keep "session time" repeating so the U-shape repeats each trading day.
    t0_wall = time.time()

    # Precompute scalers
    dt_day = params.dt / params.trading_seconds
    drift = (params.mu - 0.5 * params.sigma ** 2) * dt_day
    diff_scale = params.sigma * math.sqrt(dt_day)
    jump_prob = params.jump_lambda * dt_day

    while True:
        ts = time.time()
        # Position within the repeating trading session
        t_session = (ts - t0_wall) % params.trading_seconds
        t_norm = t_session / params.trading_seconds  # 0..1

        # ----- Price: jump-diffusion log-return -----
        z = rng.standard_normal()
        has_jump = rng.random() < jump_prob
        J = rng.normal(params.jump_mu, params.jump_sigma) if has_jump else 0.0
        r = drift + diff_scale * z + J
        S *= math.exp(r)

        # ----- Volume: seasonality + AR(1) + volatility and jump sensitivity -----
        season = u_shape_profile(t_norm)
        eps_v = rng.normal(0.0, params.sigma_v)
        logV = season + params.phi * logV + params.beta * abs(r) + (params.gamma if has_jump else 0.0) + eps_v
        V = max(1.0, math.exp(logV))

        yield {
            "ts": ts,
            "in_session": True,
            "price": S,
            "ret": r,
            "jump": bool(has_jump),
            "volume": V
        }

        # time.sleep(params.dt)


# ---------- Command-line runners ----------

def run_stdout(params: JDParams):
    """
    Continuously print newline-delimited JSON ticks to stdout.
    """
    stream = jd_price_volume_stream(params)

    # Graceful shutdown with Ctrl+C
    stop = False
    def _handler(sig, frame):
        nonlocal stop
        stop = True
        print("\nStopping...", flush=True)
    signal.signal(signal.SIGINT, _handler)

    while not stop:
        tick = next(stream)
        print(json.dumps(tick), flush=True)


def run_websocket(params: JDParams, host: str = "0.0.0.0", port: int = 8000, path: str = "/prices"):
    from fastapi import FastAPI, WebSocket
    import uvicorn, asyncio, logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("sim")

    app = FastAPI()
    stream = jd_price_volume_stream(params)  # generator has NO time.sleep

    @app.websocket(path)
    async def prices(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                tick = next(stream)
                # ↓ Option A: send as JSON (preferred)
                await ws.send_json(tick)
                await asyncio.sleep(params.dt)
        except Exception as e:
            log.exception("WebSocket loop error")
            try:
                await ws.close()
            except Exception:
                pass

    print(f"Serving WebSocket on ws://{host}:{port}{path}")
    uvicorn.run(app, host=host, port=port, log_level="info")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time jump-diffusion stock simulator with volume.")
    p.add_argument("--mode", choices=["stdout", "websocket"], default="stdout")
    p.add_argument("--dt", type=float, default=1.0, help="Seconds between ticks.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mu", type=float, default=0.08)
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument("--jump-lambda", type=float, default=0.08)
    p.add_argument("--jump-mu", type=float, default=0.0)
    p.add_argument("--jump-sigma", type=float, default=0.03)
    p.add_argument("--v0", type=float, default=150_000.0)
    p.add_argument("--phi", type=float, default=0.85)
    p.add_argument("--beta", type=float, default=22.0)
    p.add_argument("--gamma", type=float, default=0.7)
    p.add_argument("--sigma-v", type=float, default=0.35)
    p.add_argument("--trading-seconds", type=float, default=6.5*3600)
    # websocket options
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--path", type=str, default="/prices")
    return p.parse_args()


def main():
    args = parse_args()
    params = JDParams(
        dt=args.dt,
        seed=args.seed,
        mu=args.mu,
        sigma=args.sigma,
        jump_lambda=args.jump_lambda,
        jump_mu=args.jump_mu,
        jump_sigma=args.jump_sigma,
        v0=args.v0,
        phi=args.phi,
        beta=args.beta,
        gamma=args.gamma,
        sigma_v=args.sigma_v,
        trading_seconds=args.trading_seconds,
    )

    if args.mode == "stdout":
        run_stdout(params)
    else:
        run_websocket(params, host=args.host, port=args.port, path=args.path)


if __name__ == "__main__":
    main()
