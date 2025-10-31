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

# No OHLCV aggregation is used; we render tick lines directly

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
        detect_price_anomalies_absmean3std,
        detect_volume_anomalies_absmean3std,
    )
except Exception:
    detect_price_anomalies_zscore = None
    detect_volume_anomalies_zscore = None
    detect_price_anomalies_absmean3std = None
    detect_volume_anomalies_absmean3std = None


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
# Per-method anomaly stores: { method: { "price": {ts: val}, "volume": {ts: val} } }
st.session_state.setdefault(
    "anoms",
    {
        "Z-Score": {"price": {}, "volume": {}},
        "Abs-Mean+3Std": {"price": {}, "volume": {}},
        "Isolation Forest": {"price": {}, "volume": {}},
        "Model (Prophet)": {"price": {}, "volume": {}},
    },
)


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
left_col, right_col = st.columns([1, 3])
with left_col:
    st.title("WebSocket Collector")
    st.caption("Connects to a WebSocket, collects messages, and shows live price and volume lines.")


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


with left_col:
    # Controls: URL input, detection method, start/stop
    st.session_state.ws_url = st.text_input(
        "WebSocket URL",
        value=st.session_state.ws_url or "",
        placeholder="ws://localhost:8765 or wss://example.com/stream",
    )

    anom_method = st.selectbox(
        "Anomaly detection method",
        options=["Z-Score", "Abs-Mean+3Std", "Isolation Forest", "Model (Prophet)"],
        index=0,
        help="Choose how anomalies are detected on price and volume.",
    )
    st.session_state["anom_method"] = anom_method

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Start WebSocket", type="primary", use_container_width=True, key="btn_start_left"):
            start_ws()
    with btn_col2:
        if st.button("Stop WebSocket", use_container_width=True, key="btn_stop_left"):
            stop_ws()

    # Rolling window controls (use stable Streamlit keys; avoid manual reassignment)
    st.session_state.setdefault("rolling_on", False)
    st.session_state.setdefault("rolling_n", 20)
    st.toggle(
        "Rolling window (ON = use last N ticks)",
        key="rolling_on",
        value=st.session_state.get("rolling_on", False),
        help="When ON, anomaly detection uses a rolling window of the most recent N ticks; when OFF, it uses all collected ticks.",
    )
    if st.session_state.get("rolling_on", False):
        st.selectbox(
            "Window size (ticks)",
            options=[20, 50, 100],
            index={20: 0, 50: 1, 100: 2}.get(int(st.session_state.get("rolling_n", 20) or 20), 0),
            key="rolling_n",
            help="Number of most recent ticks to include when Rolling window is ON.",
        )
    if st.session_state.get("rolling_on", False):
        st.info("Rolling window ON: detection uses only the most recent N ticks (editable above).")
    else:
        st.info("Rolling window OFF: detection uses all collected ticks.")


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


# status and counters on the left; chart on the right
with left_col:
    status_placeholder = st.empty()
    error_placeholder = st.empty()
    metric_placeholder = st.empty()
with right_col:
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

    # Two-row layout: price (top), volume bars (bottom)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.72, 0.28],
    )
    # Price line (row 1)
    fig.add_trace(
        go.Scatter(
            x=df["dt"],
            y=df["price"],
            mode="lines",
            name="Price",
            line=dict(color="#1976d2", width=2),
        ),
        row=1,
        col=1,
    )
    # Volume bars (row 2), colored by price tick direction
    try:
        dir_series = (df["price"].diff().fillna(0)).apply(lambda x: 1 if x >= 0 else -1)
    except Exception:
        dir_series = pd.Series([1] * len(df), index=df.index)
    vol_colors = dir_series.map({1: "#26a69a", -1: "#ef5350"}).tolist()
    fig.add_trace(
        go.Bar(
            x=df["dt"],
            y=df["volume"],
            marker_color=vol_colors,
            name="Volume",
            opacity=0.8,
        ),
        row=2,
        col=1,
    )

    # Z-score anomaly markers for price and volume
    price_count = 0
    vol_count = 0

    # Choose detection dataframe based on rolling window setting
    rolling_on = st.session_state.get("rolling_on", False)
    rolling_n = int(st.session_state.get("rolling_n", 20) or 20)
    dfd = df.tail(max(1, rolling_n)) if rolling_on else df

    method = st.session_state.get("anom_method", "Z-Score")
    # Ensure store exists for current method (handles older sessions missing new keys)
    try:
        st.session_state["anoms"].setdefault(method, {"price": {}, "volume": {}})
    except Exception:
        st.session_state["anoms"] = {method: {"price": {}, "volume": {}}}
    if method == "Z-Score":
        # Z-score based detection on tick-to-tick returns
        if detect_price_anomalies_zscore:
            price_flags = detect_price_anomalies_zscore(dfd["price"].tolist())
            if price_flags:
                mask = pd.Series(price_flags[: len(dfd)], index=dfd.index) == 1
                # Persist newly found anomalies
                if mask.any():
                    store = st.session_state["anoms"][method]["price"]
                    for ts_val, y_val in zip(dfd.loc[mask, "ts"], dfd.loc[mask, "price"]):
                        store[float(ts_val)] = float(y_val)

        if detect_volume_anomalies_zscore:
            vol_flags = detect_volume_anomalies_zscore(dfd["volume"].tolist())
            if vol_flags:
                mask_v = pd.Series(vol_flags[: len(dfd)], index=dfd.index) == 1
                if mask_v.any():
                    store = st.session_state["anoms"][method]["volume"]
                    for ts_val, y_val in zip(dfd.loc[mask_v, "ts"], dfd.loc[mask_v, "volume"]):
                        store[float(ts_val)] = float(y_val)
    elif method == "Abs-Mean+3Std":
        # Absolute-return threshold method: |r| > mean(|r|) + 3*std(|r|)
        if detect_price_anomalies_absmean3std:
            price_flags = detect_price_anomalies_absmean3std(dfd["price"].tolist())
            if price_flags:
                mask = pd.Series(price_flags[: len(dfd)], index=dfd.index) == 1
                if mask.any():
                    store = st.session_state["anoms"][method]["price"]
                    for ts_val, y_val in zip(dfd.loc[mask, "ts"], dfd.loc[mask, "price"]):
                        store[float(ts_val)] = float(y_val)

        if detect_volume_anomalies_absmean3std:
            vol_flags = detect_volume_anomalies_absmean3std(dfd["volume"].tolist())
            if vol_flags:
                mask_v = pd.Series(vol_flags[: len(dfd)], index=dfd.index) == 1
                if mask_v.any():
                    store = st.session_state["anoms"][method]["volume"]
                    for ts_val, y_val in zip(dfd.loc[mask_v, "ts"], dfd.loc[mask_v, "volume"]):
                        store[float(ts_val)] = float(y_val)
    elif method == "Isolation Forest":
        # Apply Isolation Forest independently on price and volume only when enough data exists
        if len(dfd) >= 10:
            try:
                import numpy as np  # type: ignore
                from sklearn.ensemble import IsolationForest  # type: ignore
            except Exception:
                error_placeholder.info("Isolation Forest not available. Install scikit-learn to enable it.")
            else:
                # Price IF
                p_vals = dfd["price"].to_numpy(dtype=float)
                if p_vals.size:
                    p_mu = float(np.mean(p_vals))
                    p_sigma = float(np.std(p_vals)) or 1.0
                    p_feat = ((p_vals - p_mu) / p_sigma).reshape(-1, 1)
                    try:
                        p_if = IsolationForest(n_estimators=100, contamination=0.005, random_state=42)
                        p_if.fit(p_feat)
                        p_pred = p_if.predict(p_feat)  # -1 outlier
                        p_mask = p_pred == -1
                    except Exception:
                        p_mask = np.zeros(len(p_feat), dtype=bool)
                    if p_mask.any():
                        p_store = st.session_state["anoms"][method]["price"]
                        for ts_val, p_val, is_anom in zip(dfd["ts"], dfd["price"], p_mask):
                            if is_anom:
                                p_store[float(ts_val)] = float(p_val)

                # Volume IF
                v_vals = dfd["volume"].to_numpy(dtype=float)
                if v_vals.size:
                    v_mu = float(np.mean(v_vals))
                    v_sigma = float(np.std(v_vals)) or 1.0
                    v_feat = ((v_vals - v_mu) / v_sigma).reshape(-1, 1)
                    try:
                        v_if = IsolationForest(n_estimators=100, contamination=0.005, random_state=42)
                        v_if.fit(v_feat)
                        v_pred = v_if.predict(v_feat)
                        v_mask = v_pred == -1
                    except Exception:
                        v_mask = np.zeros(len(v_feat), dtype=bool)
                    if v_mask.any():
                        v_store = st.session_state["anoms"][method]["volume"]
                        for ts_val, v_val, is_anom in zip(dfd["ts"], dfd["volume"], v_mask):
                            if is_anom:
                                v_store[float(ts_val)] = float(v_val)
    else:
        # Prophet-based residual anomalies
        Prophet = None
        try:
            from prophet import Prophet as _Prophet  # type: ignore
            Prophet = _Prophet
        except Exception:
            try:
                from fbprophet import Prophet as _Prophet  # type: ignore
                Prophet = _Prophet
            except Exception:
                Prophet = None

        if Prophet is None:
            error_placeholder.info("Model (Prophet) not available. Install 'prophet' to enable model-based anomalies.")
        else:
            # Helper to compute residual-based flags on a single series
            def _prophet_flags(series_dt: pd.Series, series_y: pd.Series) -> pd.Series:
                min_points = 30
                if len(series_y) < min_points:
                    return pd.Series([0] * len(series_y), index=series_y.index)
                df_p = pd.DataFrame({"ds": series_dt, "y": series_y})
                try:
                    m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
                    m.fit(df_p)
                    yhat_df = m.predict(df_p[["ds"]])
                    resid = df_p["y"].values - yhat_df["yhat"].values
                    abs_resid = np.abs(resid)
                    thr = abs_resid.mean() + 3 * abs_resid.std()
                    flags = (abs_resid > thr).astype(int)
                    return pd.Series(flags, index=series_y.index)
                except Exception:
                    return pd.Series([0] * len(series_y), index=series_y.index)

            # Price flags
            try:
                import numpy as np  # local import for std/mean
            except Exception:
                np = None  # type: ignore

            if np is not None:
                p_flags = _prophet_flags(dfd["dt"], dfd["price"]) 
                if p_flags.any():
                    mask = p_flags == 1
                    store = st.session_state["anoms"][method]["price"]
                    for ts_val, y_val in zip(dfd.loc[mask, "ts"], dfd.loc[mask, "price"]):
                        store[float(ts_val)] = float(y_val)

                # Volume flags
                v_flags = _prophet_flags(dfd["dt"], dfd["volume"]) 
                if v_flags.any():
                    mask_v = v_flags == 1
                    store = st.session_state["anoms"][method]["volume"]
                    for ts_val, y_val in zip(dfd.loc[mask_v, "ts"], dfd.loc[mask_v, "volume"]):
                        store[float(ts_val)] = float(y_val)

    # After updating anomaly stores, draw all accumulated anomalies
    method_stores = st.session_state.get("anoms", {})
    current_store = method_stores.get(method, {"price": {}, "volume": {}})
    price_store = current_store.get("price", {})
    vol_store = current_store.get("volume", {})
    price_count = len(price_store)
    vol_count = len(vol_store)

    # Visual style mapping per method
    name_suffix = {
        "Z-Score": "z-score",
        "Abs-Mean+3Std": "abs+3σ",
        "Isolation Forest": "iForest",
        "Model (Prophet)": "model",
    }
    price_color = {
        "Z-Score": "#9c27b0",
        "Abs-Mean+3Std": "#9c27b0",
        "Isolation Forest": "#3f51b5",
        "Model (Prophet)": "#009688",
    }.get(method, "#9c27b0")
    vol_color = {
        "Z-Score": "#ff8f00",
        "Abs-Mean+3Std": "#ff8f00",
        "Isolation Forest": "#f57c00",
        "Model (Prophet)": "#ff7043",
    }.get(method, "#ff8f00")

    if price_store:
        p_ts = list(price_store.keys())
        p_vals = [price_store[t] for t in p_ts]
        p_dt = pd.to_datetime(p_ts, unit="s")
        fig.add_trace(
            go.Scatter(
                x=p_dt,
                y=p_vals,
                mode="markers",
                name=("Price anomaly (" + name_suffix.get(method, "z-score") + ")"),
                marker=dict(color=price_color, size=13, symbol="circle", line=dict(color="#1b1b1b", width=2)),
                hovertemplate="Price anomaly: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        for x_val in p_dt:
            fig.add_vline(x=x_val, line_color=price_color, opacity=0.15, row=1, col=1)

    if vol_store:
        v_ts = list(vol_store.keys())
        v_vals = [vol_store[t] for t in v_ts]
        v_dt = pd.to_datetime(v_ts, unit="s")
        fig.add_trace(
            go.Scatter(
                x=v_dt,
                y=v_vals,
                mode="markers",
                name=("Volume anomaly (" + name_suffix.get(method, "z-score") + ")"),
                marker=dict(color=vol_color, size=12, symbol="diamond", line=dict(color="#1b1b1b", width=2)),
                hovertemplate="Volume anomaly: %{y:.0f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        for x_val in v_dt:
            fig.add_vline(x=x_val, line_color=vol_color, opacity=0.12, line_dash="dot", row=2, col=1)

    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title=None,
        yaxis_title="Price",
        yaxis2_title="Volume",
        height=640,
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
