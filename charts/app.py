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
        rate_of_change,
        detect_anomalies_zscore,
        rate_of_change_open,
        detect_open_anomalies_zscore,
        rate_of_change_high,
        detect_high_anomalies_zscore,
        rate_of_change_low,
        detect_low_anomalies_zscore,
    )
except Exception:
    rate_of_change = None
    detect_anomalies_zscore = None
    rate_of_change_open = None
    detect_open_anomalies_zscore = None
    rate_of_change_high = None
    detect_high_anomalies_zscore = None
    rate_of_change_low = None
    detect_low_anomalies_zscore = None


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
st.session_state.setdefault("last_completed_bar_start", None)


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
st.caption("Connects to a WebSocket, collects messages, and shows OHLCV bars.")

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


# ---- OHLCV helpers ----
def _ticks_only():
    return [
        m for m in st.session_state.messages
        if isinstance(m, dict) and all(k in m for k in ("ts", "price", "volume"))
    ]


def _render_ohlcv_list():
    if ticks_to_ohlcv is None:
        bars_placeholder.info("ohlcv.py missing - cannot compute bars.")
        return
    bars = ticks_to_ohlcv(_ticks_only(), interval_s=5)
    if bars:
        # Build DataFrame and render as candlestick chart
        df = pd.DataFrame(bars)[["start", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("start")
        df["start_dt"] = pd.to_datetime(df["start"], unit="s")
        # Compute end timestamp for band highlighting (interval is 10s above)
        df["end_dt"] = pd.to_datetime(df["start"] + 10, unit="s")

        # Only include fully completed candles in the render (exclude the current in-progress bucket)
        now_dt = pd.to_datetime(int(time.time()), unit="s")
        df_completed = df[df["end_dt"] <= now_dt].copy()

        if df_completed.empty:
            bars_placeholder.info("Waiting for completed candles...")
            return

        # If no newly completed candle since last render, skip updating chart
        last_completed_start = int(df_completed["start"].max())
        if st.session_state.get("last_completed_bar_start") == last_completed_start:
            return
        st.session_state["last_completed_bar_start"] = last_completed_start

        # Work with completed bars only for plotting
        df = df_completed

        # Create subplots: price (row 1) and volume (row 2)
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.75, 0.25],
        )

        # Price candlestick on row 1
        fig.add_trace(
            go.Candlestick(
                x=df["start_dt"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
                name="OHLC",
            ),
            row=1,
            col=1,
        )

        # Volume bars on row 2, colored by direction
        try:
            dir_series = (df["close"] - df["open"]).apply(lambda x: 1 if x >= 0 else -1)
        except Exception:
            dir_series = pd.Series([1] * len(df), index=df.index)
        vol_colors = dir_series.map({1: "#26a69a", -1: "#ef5350"}).tolist()
        fig.add_trace(
            go.Bar(
                x=df["start_dt"],
                y=df["volume"],
                marker_color=vol_colors,
                name="Volume",
                opacity=0.7,
            ),
            row=2,
            col=1,
        )

        # Baseline for anomaly dots along the bottom of the chart
        y_baseline = float(df["low"].min())
        # Use small vertical offsets so multiple anomaly dots at the same x are all visible
        price_min = float(df["low"].min())
        price_max = float(df["high"].max())
        price_range = max(1e-9, price_max - price_min)
        y_offset = price_range * 0.01  # 1% of price range
        y_close_level = y_baseline
        y_open_level = y_baseline + y_offset
        y_high_level = y_baseline + (2 * y_offset)
        y_low_level = y_baseline + (3 * y_offset)

        # ---- Anomaly overlay using close-to-close returns ----
        if rate_of_change and detect_anomalies_zscore:
            closes = df["close"].tolist()
            returns = rate_of_change(closes)  # len n-1
            valid_returns = [r for r in returns if r is not None]
            flags = [0]
            if valid_returns:
                flags += detect_anomalies_zscore(valid_returns)
            if len(flags) < len(df):
                flags += [0] * (len(df) - len(flags))
            df["anomaly"] = flags[: len(df)]
            mask = df["anomaly"] == 1
            if mask.any():
                # Highlight the close price of anomalous bars with a colored marker
                anom_color = "#9c27b0"  # distinctive color for close anomalies (purple)
                x_anom = df.loc[mask, "start_dt"]
                y_anom = df.loc[mask, "close"]

                fig.add_trace(
                    go.Scatter(
                        x=x_anom,
                        y=[y_close_level] * len(x_anom),
                        customdata=y_anom,
                        mode="markers",
                        marker=dict(
                            symbol="circle",
                            size=10,
                            color=anom_color,
                            line=dict(width=1, color="#1b1b1b"),
                        ),
                        name="Close anomaly",
                        hovertemplate="Anomaly close: %{customdata:.4f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

                # Arrows removed per request; keep only bottom dots

        # ---- Anomaly overlay using open-to-open returns ----
        if rate_of_change_open and detect_open_anomalies_zscore:
            opens = df["open"].tolist()
            open_flags = detect_open_anomalies_zscore(opens)
            if len(open_flags) < len(df):
                open_flags += [0] * (len(df) - len(open_flags))
            df["anomaly_open"] = open_flags[: len(df)]
            mask_open = df["anomaly_open"] == 1
            if mask_open.any():
                open_color = "#1976d2"  # blue for open anomalies
                x_open = df.loc[mask_open, "start_dt"]
                y_open = df.loc[mask_open, "open"]

                fig.add_trace(
                    go.Scatter(
                        x=x_open,
                        y=[y_open_level] * len(x_open),
                        customdata=y_open,
                        mode="markers",
                        marker=dict(
                            symbol="circle",
                            size=10,
                            color=open_color,
                            line=dict(width=1, color="#1b1b1b"),
                        ),
                        name="Open anomaly",
                        hovertemplate="Anomaly open: %{customdata:.4f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

                # Arrows removed per request; keep only bottom dots

        # ---- Anomaly overlay using high-to-high returns ----
        if rate_of_change_high and detect_high_anomalies_zscore:
            highs = df["high"].tolist()
            high_flags = detect_high_anomalies_zscore(highs)
            if len(high_flags) < len(df):
                high_flags += [0] * (len(df) - len(high_flags))
            df["anomaly_high"] = high_flags[: len(df)]
            mask_high = df["anomaly_high"] == 1
            if mask_high.any():
                high_color = "#ff8f00"  # orange for high anomalies
                x_high = df.loc[mask_high, "start_dt"]
                y_high = df.loc[mask_high, "high"]

                fig.add_trace(
                    go.Scatter(
                        x=x_high,
                        y=[y_high_level] * len(x_high),
                        customdata=y_high,
                        mode="markers",
                        marker=dict(
                            symbol="circle",
                            size=10,
                            color=high_color,
                            line=dict(width=1, color="#1b1b1b"),
                        ),
                        name="High anomaly",
                        hovertemplate="Anomaly high: %{customdata:.4f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

        # Low anomalies

        # ---- Anomaly overlay using low-to-low returns ----
        if rate_of_change_low and detect_low_anomalies_zscore:
            lows = df["low"].tolist()
            low_flags = detect_low_anomalies_zscore(lows)
            if len(low_flags) < len(df):
                low_flags += [0] * (len(df) - len(low_flags))
            df["anomaly_low"] = low_flags[: len(df)]
            mask_low = df["anomaly_low"] == 1
            if mask_low.any():
                low_color = "#43a047"  # green for low anomalies
                x_low = df.loc[mask_low, "start_dt"]
                y_low = df.loc[mask_low, "low"]

                fig.add_trace(
                    go.Scatter(
                        x=x_low,
                        y=[y_low_level] * len(x_low),
                        customdata=y_low,
                        mode="markers",
                        marker=dict(
                            symbol="circle",
                            size=10,
                            color=low_color,
                            line=dict(width=1, color="#1b1b1b"),
                        ),
                        name="Low anomaly",
                        hovertemplate="Anomaly low: %{customdata:.4f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

        # Axes and layout
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig.update_layout(
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Time",
            yaxis_title="Price",
            yaxis2_title="Volume",
            height=720,
            legend_tracegroupgap=6,
        )
        bars_placeholder.plotly_chart(fig, use_container_width=True)
    else:
        bars_placeholder.info("No OHLCV bars yet - collect more ticks.")


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
