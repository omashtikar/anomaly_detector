# app.py
import argparse
import json
import queue
import sys
import threading
import time

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from ohlcv import ticks_to_ohlcv
except Exception:
    ticks_to_ohlcv = None

try:
    from websocket import WebSocketApp
except ImportError:
    st.error("Missing dependency: install with `pip install websocket-client`")
    st.stop()

# Anomaly utils
try:
    from utils import (
        detect_price_anomalies_zscore,
        detect_volume_anomalies_zscore,
    )
except Exception:
    detect_price_anomalies_zscore = None
    detect_volume_anomalies_zscore = None


# ----- CLI args -----
def get_cli_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ws", "--websocket", dest="ws", default=None, help="WebSocket URL")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


args = get_cli_args()

# ----- Session state -----
st.session_state.setdefault("messages", [])
st.session_state.setdefault("ws_queue", queue.Queue())
st.session_state.setdefault("ws_thread", None)
st.session_state.setdefault("ws_stop_event", threading.Event())
st.session_state.setdefault("ws_url", args.ws)
st.session_state.setdefault("ws_status", "disconnected")
st.session_state.setdefault("ws_error", None)


# ----- WebSocket worker -----
def ws_worker(url: str, out_q: queue.Queue, stop_event: threading.Event):
    def on_open(ws):
        out_q.put({"_event": "open"})

    def on_message(ws, message):
        try:
            out_q.put(json.loads(message))
        except Exception:
            out_q.put(message)

    def on_error(ws, error):
        out_q.put({"_event": "error", "error": str(error)})

    def on_close(ws, status_code, close_msg):
        out_q.put({"_event": "close", "status_code": status_code, "reason": close_msg})

    ws = WebSocketApp(url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
    t = threading.Thread(target=lambda: ws.run_forever(ping_interval=25, ping_timeout=10), daemon=True)
    t.start()

    # keep thread alive until asked to stop
    while t.is_alive() and not stop_event.is_set():
        time.sleep(0.2)
    try:
        ws.close()
    except Exception:
        pass


# ----- UI -----
st.title("WebSocket Collector")
st.caption("Connects to a WebSocket, collects messages, and shows live price and volume lines.")

st.session_state.ws_url = st.text_input(
    "WebSocket URL",
    value=st.session_state.ws_url or "",
    placeholder="ws://localhost:8765 or wss://example.com/stream",
)

col1, col2 = st.columns(2)


def start_ws():
    if not st.session_state.ws_url:
        st.session_state.ws_error = "Please provide a WebSocket URL."
        return
    # reset
    st.session_state.ws_error = None
    st.session_state.messages.clear()
    st.session_state.ws_stop_event = threading.Event()
    st.session_state.ws_thread = threading.Thread(
        target=ws_worker,
        args=(st.session_state.ws_url, st.session_state.ws_queue, st.session_state.ws_stop_event),
        daemon=True,
    )
    st.session_state.ws_thread.start()
    st.session_state.ws_status = "connecting"


def stop_ws():
    if st.session_state.ws_thread and st.session_state.ws_thread.is_alive():
        st.session_state.ws_stop_event.set()
    st.session_state.ws_status = "disconnected"


with col1:
    if st.button("Start WebSocket", type="primary", use_container_width=True):
        start_ws()
with col2:
    if st.button("Stop WebSocket", use_container_width=True):
        stop_ws()


# drain queue helper
def drain_queue():
    updated = 0
    while True:
        try:
            item = st.session_state.ws_queue.get_nowait()
        except queue.Empty:
            break
        updated += 1
        if isinstance(item, dict) and item.get("_event") == "open":
            st.session_state.ws_status = "connected"
        elif isinstance(item, dict) and item.get("_event") == "error":
            st.session_state.ws_error = item.get("error")
            st.session_state.ws_status = "error"
        elif isinstance(item, dict) and item.get("_event") == "close":
            st.session_state.ws_status = "closed"
        else:
            st.session_state.messages.append(item)
    return updated


# status row
status_placeholder = st.empty()
error_placeholder = st.empty()
metric_placeholder = st.empty()
bars_placeholder = st.empty()


# ---- Tick helpers ----
def _ticks_only():
    return [
        m for m in st.session_state.messages
        if isinstance(m, dict) and all(k in m for k in ("ts", "price", "volume"))
    ]


def _render_ohlcv_list():
    # Render a simple live line chart for price and volume as ticks arrive
    ticks = _ticks_only()
    if not ticks:
        bars_placeholder.info("Waiting for tick data...")
        return

    df = pd.DataFrame(ticks)[["ts", "price", "volume"]]
    df = df.sort_values("ts")
    df["dt"] = pd.to_datetime(df["ts"], unit="s")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Price line
    fig.add_trace(
        go.Scatter(
            x=df["dt"],
            y=df["price"],
            mode="lines",
            name="Price",
            line=dict(color="#1976d2", width=2),
        ),
        secondary_y=False,
    )
    # Volume line (secondary axis)
    fig.add_trace(
        go.Scatter(
            x=df["dt"],
            y=df["volume"],
            mode="lines",
            name="Volume",
            line=dict(color="#8e24aa", width=1),
        ),
        secondary_y=True,
    )

    # Z-score anomaly markers for price and volume
    price_count = 0
    vol_count = 0

    if detect_price_anomalies_zscore:
        price_flags = detect_price_anomalies_zscore(df["price"].tolist())
        if price_flags:
            mask = pd.Series(price_flags[: len(df)]) == 1
            if mask.any():
                price_count = int(mask.sum())
                fig.add_trace(
                    go.Scatter(
                        x=df.loc[mask, "dt"],
                        y=df.loc[mask, "price"],
                        mode="markers",
                        name="Price anomaly",
                        marker=dict(color="#9c27b0", size=11, symbol="circle", line=dict(color="#1b1b1b", width=1)),
                        hovertemplate="Price anomaly: %{y:.4f}<extra></extra>",
                    ),
                    secondary_y=False,
                )
                # Add subtle vlines at anomaly timestamps for visibility
                for x_val in df.loc[mask, "dt"]:
                    fig.add_vline(x=x_val, line_color="#9c27b0", opacity=0.15)

    if detect_volume_anomalies_zscore:
        vol_flags = detect_volume_anomalies_zscore(df["volume"].tolist())
        if vol_flags:
            mask_v = pd.Series(vol_flags[: len(df)]) == 1
            if mask_v.any():
                vol_count = int(mask_v.sum())
                fig.add_trace(
                    go.Scatter(
                        x=df.loc[mask_v, "dt"],
                        y=df.loc[mask_v, "volume"],
                        mode="markers",
                        name="Volume anomaly",
                        marker=dict(color="#ff8f00", size=10, symbol="diamond", line=dict(color="#1b1b1b", width=1)),
                        hovertemplate="Volume anomaly: %{y:.0f}<extra></extra>",
                    ),
                    secondary_y=True,
                )
                for x_val in df.loc[mask_v, "dt"]:
                    fig.add_vline(x=x_val, line_color="#ff8f00", opacity=0.12, line_dash="dot")

    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Time",
        yaxis_title="Price",
        yaxis2_title="Volume",
        height=520,
        legend_tracegroupgap=6,
    )
    bars_placeholder.plotly_chart(fig, use_container_width=True)

    # Streamlit metrics for quick anomaly visibility
    try:
        c1, c2 = metric_placeholder.columns(2)
        c1.metric(label="Price anomalies", value=str(price_count))
        c2.metric(label="Volume anomalies", value=str(vol_count))
    except Exception:
        metric_placeholder.write(f"Price anomalies: {price_count} | Volume anomalies: {vol_count}")


# live update toggle
live = st.toggle("Live update", value=True, help="Continuously refresh the message count.")

# initial render
drain_queue()
status_placeholder.write(f"Status: **{st.session_state.ws_status}**")
if st.session_state.ws_error:
    error_placeholder.error(st.session_state.ws_error)
_render_ohlcv_list()

# live loop (no extra packages; updates UI ~2x/sec)
if live and st.session_state.ws_status in {"connecting", "connected"}:
    # Run a short-lived loop so we don't block forever on a single run.
    # On every rerun, this block can continue updating while the WebSocket thread feeds data.
    for _ in range(1200):  # ~10 minutes total; rerun or toggle again to extend
        changed = drain_queue()
        if changed:
            _render_ohlcv_list()
            status_placeholder.write(f"Status: **{st.session_state.ws_status}**")
            if st.session_state.ws_error:
                error_placeholder.error(st.session_state.ws_error)
        time.sleep(0.5)

# Ensure the app uses wide layout so the chart can take more space
try:
    import streamlit as st  # type: ignore
    # set_page_config must be called before any other st.* call
    try:
        st.set_page_config(layout="wide")
    except Exception:
        # Already configured or running in a non-Streamlit context
        pass
except Exception:
    pass
