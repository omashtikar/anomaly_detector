# realtime_candles.py
# Live candlestick chart from a continuous tick stream (WebSocket or local generator).
# Requires: pip install streamlit plotly pandas numpy websockets fastapi uvicorn

import math
import time
import json
import asyncio
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------------------- Jump-diffusion tick generator (local mode) ----------------------

def u_shape_profile(t_norm, a=0.6, b=0.6):
    base = (t_norm**a) * ((1 - t_norm)**b)
    return 0.5 + 0.5 * (base / 0.25)

class JDParams:
    def __init__(self,
                 s0=100.0, mu=0.08, sigma=0.25,
                 jump_lambda=0.08, jump_mu=0.0, jump_sigma=0.03,
                 v0=150_000, phi=0.85, beta=22.0, gamma=0.7, sigma_v=0.35,
                 dt=0.5, trading_seconds=6.5*3600, seed=42):
        self.s0=s0; self.mu=mu; self.sigma=sigma
        self.jump_lambda=jump_lambda; self.jump_mu=jump_mu; self.jump_sigma=jump_sigma
        self.v0=v0; self.phi=phi; self.beta=beta; self.gamma=gamma; self.sigma_v=sigma_v
        self.dt=dt; self.trading_seconds=trading_seconds; self.seed=seed

def jd_price_volume_stream(params: JDParams):
    rng = np.random.default_rng(params.seed)
    S = float(params.s0)
    logV = math.log(max(params.v0, 1.0))
    t0_wall = time.time()

    dt_day   = params.dt / params.trading_seconds
    drift    = (params.mu - 0.5*params.sigma**2) * dt_day
    diff_sc  = params.sigma * math.sqrt(dt_day)
    jump_p   = params.jump_lambda * dt_day

    while True:
        ts = time.time()
        t_session = (ts - t0_wall) % params.trading_seconds
        t_norm = t_session / params.trading_seconds

        z = rng.standard_normal()
        has_jump = rng.random() < jump_p
        J = rng.normal(params.jump_mu, params.jump_sigma) if has_jump else 0.0
        r = drift + diff_sc*z + J
        S *= math.exp(r)

        season = u_shape_profile(t_norm)
        eps_v = rng.normal(0.0, params.sigma_v)
        logV = season + params.phi*logV + params.beta*abs(r) + (params.gamma if has_jump else 0.0) + eps_v
        V = max(1.0, math.exp(logV))

        yield {
            "ts": float(ts),
            "price": float(S),
            "ret": float(r),
            "jump": bool(has_jump),
            "volume": float(V),
        }
        # NOTE: no time.sleep here; Streamlit's loop handles pacing

# ---------------------- WebSocket receiver (pull one message per iteration) ----------------------

async def ws_recv_one(ws_url: str, timeout: float = 2.0):
    """
    Connect if needed, receive exactly one tick (JSON), then return it.
    Short-lived connections keep Streamlit responsive and simple.
    """
    import websockets
    try:
        async with websockets.connect(ws_url, ping_interval=None, close_timeout=1.0) as ws:
            msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
            return json.loads(msg)
    except Exception as e:
        # Return None on failure; the caller can show a warning
        return None

def recv_tick_ws(ws_url: str, timeout: float = 2.0):
    return asyncio.run(ws_recv_one(ws_url, timeout=timeout))

# ---------------------- OHLC aggregator ----------------------

def update_ohlc(df_ticks: pd.DataFrame, bar_secs: int) -> pd.DataFrame:
    """
    Convert a ticks DataFrame (ts, price, volume) into OHLCV bars by bar_secs.
    """
    if df_ticks.empty:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

    # Bucket by floor(ts / bar_secs) * bar_secs
    buckets = (np.floor(df_ticks["ts"].values / bar_secs) * bar_secs).astype(np.int64)
    df_ticks = df_ticks.copy()
    df_ticks["bucket_ts"] = buckets
    df_ticks["bucket_time"] = pd.to_datetime(df_ticks["bucket_ts"], unit="s")

    # Aggregations
    o = df_ticks.groupby("bucket_time")["price"].first()
    h = df_ticks.groupby("bucket_time")["price"].max()
    l = df_ticks.groupby("bucket_time")["price"].min()
    c = df_ticks.groupby("bucket_time")["price"].last()
    v = df_ticks.groupby("bucket_time")["volume"].sum()

    bars = pd.concat([o.rename("open"),
                      h.rename("high"),
                      l.rename("low"),
                      c.rename("close"),
                      v.rename("volume")], axis=1).sort_index()
    return bars

# ---------------------- Streamlit UI ----------------------

st.set_page_config(page_title="Real-time Candles", layout="wide")
st.title("📈 Real-time Candlestick (from continuous ticks)")

with st.sidebar:
    st.subheader("Source & Settings")
    mode = st.radio("Data source", ["WebSocket (from sim_feed.py)", "Local generator"], index=0)
    ws_url = st.text_input("WebSocket URL", value="ws://127.0.0.1:8000/prices",
                           help="Run: python sim_feed.py --mode websocket --host 127.0.0.1 --port 8000")
    bar_secs = st.number_input("Candle period (seconds)", 1, 300, 5, 1)
    refresh = st.number_input("UI refresh interval (seconds)", 0.2, 5.0, 1.0, 0.1)
    max_bars = st.number_input("Max bars to display", 50, 2000, 300, 10)
    st.caption("Tip: lower refresh for smoother updates; increase candle period for less noise.")

# State: rolling tick buffer
if "ticks" not in st.session_state:
    st.session_state.ticks = []  # list of dicts

# Prepare local generator (if needed)
if mode == "Local generator":
    if "gen" not in st.session_state:
        st.session_state.gen = jd_price_volume_stream(JDParams(dt=refresh))
else:
    st.session_state.gen = None

chart_placeholder = st.empty()
vol_placeholder = st.empty()
status = st.empty()

# Main loop (runs until you stop the app)
while True:
    # 1) Pull exactly one tick
    if mode == "Local generator":
        tick = next(st.session_state.gen)
        # pace the local generator with refresh, since it doesn't sleep
        time.sleep(float(refresh))
    else:
        tick = recv_tick_ws(ws_url, timeout=max(1.0, float(refresh) + 0.5))
        # throttle refresh loop when using WS
        time.sleep(float(refresh))

    if tick is None:
        status.warning("No tick received (WS not connected?). Check the URL and that sim_feed.py is running.")
        continue

    # 2) Append & trim tick buffer
    st.session_state.ticks.append({"ts": float(tick["ts"]),
                                   "price": float(tick["price"]),
                                   "volume": float(tick.get("volume", 0.0))})
    # Keep last N seconds worth of ticks ~ max_bars * bar_secs * a small factor
    max_ticks = int(max_bars * (bar_secs * 2))
    if len(st.session_state.ticks) > max_ticks:
        st.session_state.ticks = st.session_state.ticks[-max_ticks:]

    # 3) Build OHLCV bars
    df_ticks = pd.DataFrame.from_records(st.session_state.ticks)
    bars = update_ohlc(df_ticks, int(bar_secs))
    if bars.empty:
        continue

    if len(bars) > max_bars:
        bars = bars.iloc[-max_bars:]

    # 4) Plot candlesticks with Plotly
    fig = go.Figure(data=[
        go.Candlestick(
            x=bars.index,
            open=bars["open"], high=bars["high"],
            low=bars["low"], close=bars["close"],
            name="Price"
        )
    ])
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Price",
        margin=dict(l=10, r=10, t=30, b=10),
        height=520,
        xaxis_rangeslider_visible=False,
    )

    chart_placeholder.plotly_chart(fig, use_container_width=True)

    # Optional: simple volume plot (separate)
    vol_fig = go.Figure(data=[
        go.Bar(x=bars.index, y=bars["volume"], name="Volume")
    ])
    vol_fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Volume",
        margin=dict(l=10, r=10, t=10, b=10),
        height=180,
    )
    vol_placeholder.plotly_chart(vol_fig, use_container_width=True)

    # 5) Status line
    last = bars.iloc[-1]
    status.markdown(
        f"**Last bar** {bars.index[-1]} — O {last.open:.2f} | H {last.high:.2f} | "
        f"L {last.low:.2f} | C {last.close:.2f} | Vol {int(last.volume):,}"
    )
