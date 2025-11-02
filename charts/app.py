import queue
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional
import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from exchange_server.sim_feed import JDParams, jd_price_volume_stream
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from exchange_server.sim_feed import JDParams, jd_price_volume_stream

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

st.set_page_config(layout="wide")

ANOM_METHODS = [
    "Z-Score",
    "Abs-Mean+3Std",
    "Isolation Forest",
    "Model (Prophet)",
]
DEFAULT_PARAMS = JDParams()


def fresh_anom_store() -> Dict[str, Dict[str, Dict[float, float]]]:
    return {method: {"price": {}, "volume": {}} for method in ANOM_METHODS}


if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "feed_queue" not in st.session_state:
    st.session_state["feed_queue"] = queue.Queue()
if "sim_thread" not in st.session_state:
    st.session_state["sim_thread"] = None
if "sim_stop_event" not in st.session_state:
    st.session_state["sim_stop_event"] = threading.Event()
if "sim_status" not in st.session_state:
    st.session_state["sim_status"] = "stopped"
if "sim_error" not in st.session_state:
    st.session_state["sim_error"] = None
if "sim_params" not in st.session_state:
    st.session_state["sim_params"] = asdict(DEFAULT_PARAMS)
if "sim_runtime_params" not in st.session_state:
    st.session_state["sim_runtime_params"] = None
if "anoms" not in st.session_state:
    st.session_state["anoms"] = fresh_anom_store()
if "max_points" not in st.session_state:
    st.session_state["max_points"] = 2000
st.session_state.setdefault("anom_method", ANOM_METHODS[0])
st.session_state.setdefault("anom_method_select", st.session_state["anom_method"])
st.session_state.setdefault("rolling_on", False)
st.session_state.setdefault("rolling_n", 20)
st.session_state.setdefault("refresh_interval", 0.5)
st.session_state.setdefault("chart_rendered", False)
st.session_state.setdefault("chart_render_count", 0)


def build_params_from_state() -> JDParams:
    params_dict: Dict[str, Any] = dict(st.session_state["sim_params"])
    seed_val = params_dict.get("seed")
    if seed_val in ("", None):
        params_dict["seed"] = None
    else:
        try:
            params_dict["seed"] = int(seed_val)
        except (TypeError, ValueError):
            params_dict["seed"] = None
    return JDParams(**params_dict)


def sim_worker(
    params: JDParams,
    out_q: "queue.Queue[Dict[str, Any]]",
    stop_event: threading.Event,
) -> None:
    """Background generator that streams ticks into a queue."""
    try:
        stream = jd_price_volume_stream(params)
        out_q.put({"_event": "start", "params": asdict(params)})
        while not stop_event.is_set():
            tick = next(stream)
            out_q.put(tick)
            sleep_for = max(0.0, float(params.dt))
            if sleep_for > 0.0:
                time.sleep(sleep_for)
    except Exception as exc:
        out_q.put({"_event": "error", "error": str(exc)})
    finally:
        out_q.put({"_event": "stopped"})


def start_sim() -> None:
    drain_queue()
    existing = st.session_state.get("sim_thread")
    stop_event = st.session_state.get("sim_stop_event")
    if existing and existing.is_alive():
        if stop_event and stop_event.is_set():
            st.session_state["sim_thread"] = None
        else:
            return
    else:
        st.session_state["sim_thread"] = None

    st.session_state["sim_error"] = None
    st.session_state["messages"] = []
    st.session_state["anoms"] = fresh_anom_store()
    st.session_state["feed_queue"] = queue.Queue()
    st.session_state["chart_rendered"] = False

    stop_event = threading.Event()
    st.session_state["sim_stop_event"] = stop_event

    params = build_params_from_state()
    st.session_state["sim_runtime_params"] = asdict(params)

    thread = threading.Thread(
        target=sim_worker,
        args=(params, st.session_state["feed_queue"], stop_event),
        daemon=True,
        name="jd-sim-feed",
    )
    st.session_state["sim_thread"] = thread
    st.session_state["sim_status"] = "starting"
    thread.start()


def stop_sim() -> None:
    thread = st.session_state.get("sim_thread")
    if thread and thread.is_alive():
        st.session_state["sim_stop_event"].set()
        st.session_state["sim_status"] = "stopping"
    else:
        st.session_state["sim_status"] = "stopped"
        st.session_state["sim_thread"] = None
    st.session_state["chart_rendered"] = False


def drain_queue() -> int:
    updated = 0
    max_points = int(st.session_state.get("max_points", 0) or 0)
    q = st.session_state.get("feed_queue")
    if q is None:
        return updated

    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            break

        updated += 1
        if isinstance(item, dict) and item.get("_event") == "start":
            st.session_state["sim_status"] = "running"
            st.session_state["sim_runtime_params"] = item.get("params") or st.session_state.get("sim_runtime_params")
        elif isinstance(item, dict) and item.get("_event") == "error":
            st.session_state["sim_error"] = item.get("error")
            st.session_state["sim_status"] = "error"
        elif isinstance(item, dict) and item.get("_event") == "stopped":
            st.session_state["sim_status"] = "stopped"
            st.session_state["sim_thread"] = None
        else:
            st.session_state["messages"].append(item)
            if max_points and len(st.session_state["messages"]) > max_points:
                st.session_state["messages"] = st.session_state["messages"][-max_points:]
    return updated


def _ticks_only():
    return [
        m
        for m in st.session_state["messages"]
        if isinstance(m, dict) and all(k in m for k in ("ts", "price", "volume"))
    ]


def update_anomaly_store_for_df(
    df: pd.DataFrame,
    method: str,
    rolling_on: bool,
    rolling_n: int,
    notice_placeholder: Optional["st.delta_generator.DeltaGenerator"] = None,
    reset: bool = False,
):
    st.session_state.setdefault("anoms", fresh_anom_store())
    if reset or method not in st.session_state["anoms"]:
        st.session_state["anoms"][method] = {"price": {}, "volume": {}}

    store = st.session_state["anoms"][method]
    price_store = store["price"]
    vol_store = store["volume"]

    if reset:
        price_store.clear()
        vol_store.clear()

    if df.empty:
        return price_store, vol_store

    window_n = int(rolling_n or 1)
    window_n = max(1, window_n)
    dfd = df.tail(window_n) if rolling_on else df

    if method == "Z-Score":
        if detect_price_anomalies_zscore:
            price_flags = detect_price_anomalies_zscore(dfd["price"].tolist())
            if price_flags:
                mask = pd.Series(price_flags[: len(dfd)], index=dfd.index) == 1
                if mask.any():
                    for ts_val, y_val in zip(dfd.loc[mask, "ts"], dfd.loc[mask, "price"]):
                        price_store[float(ts_val)] = float(y_val)

        if detect_volume_anomalies_zscore:
            vol_flags = detect_volume_anomalies_zscore(dfd["volume"].tolist())
            if vol_flags:
                mask_v = pd.Series(vol_flags[: len(dfd)], index=dfd.index) == 1
                if mask_v.any():
                    for ts_val, y_val in zip(dfd.loc[mask_v, "ts"], dfd.loc[mask_v, "volume"]):
                        vol_store[float(ts_val)] = float(y_val)

    elif method == "Abs-Mean+3Std":
        if detect_price_anomalies_absmean3std:
            price_flags = detect_price_anomalies_absmean3std(dfd["price"].tolist())
            if price_flags:
                mask = pd.Series(price_flags[: len(dfd)], index=dfd.index) == 1
                if mask.any():
                    for ts_val, y_val in zip(dfd.loc[mask, "ts"], dfd.loc[mask, "price"]):
                        price_store[float(ts_val)] = float(y_val)

        if detect_volume_anomalies_absmean3std:
            vol_flags = detect_volume_anomalies_absmean3std(dfd["volume"].tolist())
            if vol_flags:
                mask_v = pd.Series(vol_flags[: len(dfd)], index=dfd.index) == 1
                if mask_v.any():
                    for ts_val, y_val in zip(dfd.loc[mask_v, "ts"], dfd.loc[mask_v, "volume"]):
                        vol_store[float(ts_val)] = float(y_val)

    elif method == "Isolation Forest":
        if len(dfd) >= 10:
            try:
                import numpy as np  # type: ignore
                from sklearn.ensemble import IsolationForest  # type: ignore
            except Exception:
                if notice_placeholder is not None:
                    notice_placeholder.info("Isolation Forest not available. Install scikit-learn to enable it.")
            else:
                p_vals = dfd["price"].to_numpy(dtype=float)
                if p_vals.size:
                    p_mu = float(np.mean(p_vals))
                    p_sigma = float(np.std(p_vals)) or 1.0
                    p_feat = ((p_vals - p_mu) / p_sigma).reshape(-1, 1)
                    try:
                        p_if = IsolationForest(n_estimators=100, contamination=0.005, random_state=42)
                        p_if.fit(p_feat)
                        p_pred = p_if.predict(p_feat)
                        p_mask = p_pred == -1
                    except Exception:
                        p_mask = np.zeros(len(p_feat), dtype=bool)
                    if p_mask.any():
                        for ts_val, p_val, is_anom in zip(dfd["ts"], dfd["price"], p_mask):
                            if is_anom:
                                price_store[float(ts_val)] = float(p_val)

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
                        for ts_val, v_val, is_anom in zip(dfd["ts"], dfd["volume"], v_mask):
                            if is_anom:
                                vol_store[float(ts_val)] = float(v_val)
        else:
            if notice_placeholder is not None:
                notice_placeholder.info("Isolation Forest needs at least 10 ticks to warm up.")

    else:
        try:
            from prophet import Prophet as _Prophet  # type: ignore
        except Exception:
            try:
                from fbprophet import Prophet as _Prophet  # type: ignore
            except Exception:
                _Prophet = None

        if _Prophet is None:
            if notice_placeholder is not None:
                notice_placeholder.info("Model (Prophet) not available. Install 'prophet' to enable it.")
        else:
            def _prophet_flags(series_dt: pd.Series, series_y: pd.Series) -> pd.Series:
                min_points = 30
                if len(series_y) < min_points:
                    return pd.Series([0] * len(series_y), index=series_y.index)
                df_p = pd.DataFrame({"ds": series_dt, "y": series_y})
                try:
                    m = _Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
                    m.fit(df_p)
                    yhat_df = m.predict(df_p[["ds"]])
                    resid = df_p["y"].values - yhat_df["yhat"].values
                    import numpy as np  # type: ignore

                    abs_resid = np.abs(resid)
                    thr = abs_resid.mean() + 3 * abs_resid.std()
                    flags = (abs_resid > thr).astype(int)
                    return pd.Series(flags, index=series_y.index)
                except Exception:
                    return pd.Series([0] * len(series_y), index=series_y.index)

            try:
                import numpy as np  # type: ignore
            except Exception:
                np = None  # type: ignore

            if np is not None:
                p_flags = _prophet_flags(dfd["dt"], dfd["price"])
                if p_flags.any():
                    mask = p_flags == 1
                    for ts_val, y_val in zip(dfd.loc[mask, "ts"], dfd.loc[mask, "price"]):
                        price_store[float(ts_val)] = float(y_val)

                v_flags = _prophet_flags(dfd["dt"], dfd["volume"])
                if v_flags.any():
                    mask_v = v_flags == 1
                    for ts_val, y_val in zip(dfd.loc[mask_v, "ts"], dfd.loc[mask_v, "volume"]):
                        vol_store[float(ts_val)] = float(y_val)

    return price_store, vol_store


def recompute_full_anomalies(method: str) -> None:
    st.session_state.setdefault("anoms", fresh_anom_store())
    ticks = _ticks_only()
    if not ticks:
        st.session_state["anoms"][method] = {"price": {}, "volume": {}}
        return

    df = pd.DataFrame(ticks)[["ts", "price", "volume"]]
    df = df.sort_values("ts")
    df["dt"] = pd.to_datetime(df["ts"], unit="s")
    update_anomaly_store_for_df(
        df,
        method,
        rolling_on=False,
        rolling_n=len(df),
        notice_placeholder=None,
        reset=True,
    )


params = st.session_state["sim_params"]

left_col, right_col = st.columns([1, 3])
with left_col:
    st.title("Anomaly Dashboard")

    with st.container():
        anom_method = st.selectbox(
            "Anomaly detection method",
            options=ANOM_METHODS,
            key="anom_method_select",
            help="Choose the algorithm used to highlight price and volume anomalies.",
        )

        current_method = st.session_state.get("anom_method", ANOM_METHODS[0])
        if anom_method != current_method:
            st.session_state["anom_method"] = anom_method
            stop_sim()
            drain_queue()
            with st.spinner("Recomputing anomalies..."):
                recompute_full_anomalies(anom_method)
            st.session_state["chart_rendered"] = False

        btn_col1, btn_col2 = st.columns(2)
        if btn_col1.button("Start simulator", type="primary", use_container_width=True):
            start_sim()
        if btn_col2.button("Stop simulator", use_container_width=True):
            stop_sim()

        sim_status = st.session_state.get("sim_status", "stopped")
        if sim_status == "stopped":
            st.caption("Simulator paused. Click Start simulator to stream new ticks.")
        elif sim_status == "error":
            st.caption("Simulator error. Adjust parameters and click Start simulator to try again.")

        max_points_input = st.number_input(
            "Max ticks to keep",
            min_value=200,
            max_value=10000,
            step=200,
            value=int(st.session_state["max_points"]),
            help="Older ticks are dropped once this limit is reached.",
        )
        st.session_state["max_points"] = int(max_points_input)

        st.toggle(
            "Rolling window (ON = use last N ticks)",
            key="rolling_on",
            value=st.session_state.get("rolling_on", False),
            help="When ON, anomaly detection uses the most recent N ticks.",
        )
        if st.session_state.get("rolling_on", False):
            st.selectbox(
                "Window size (ticks)",
                options=[20, 50, 100],
                index={20: 0, 50: 1, 100: 2}.get(int(st.session_state.get("rolling_n", 20) or 20), 0),
                key="rolling_n",
                help="Number of ticks considered when the rolling window is enabled.",
            )

        refresh_val = st.number_input(
            "Screen refresh rate (seconds)",
            min_value=0.1,
            max_value=5.0,
            step=0.1,
            value=float(st.session_state.get("refresh_interval", 0.5)),
            help="Controls how often the chart refreshes while the simulator runs.",
        )
        st.session_state["refresh_interval"] = float(refresh_val)

    status_placeholder = st.empty()
    sim_error_placeholder = st.empty()
    notice_placeholder = st.empty()
    metric_placeholder = st.empty()

    with st.expander("Simulator parameters", expanded=False):
        col_dt, col_seed = st.columns(2)
        params["dt"] = float(
            col_dt.number_input(
                "Seconds between ticks",
                min_value=0.01,
                max_value=5.0,
                step=0.05,
                value=float(params["dt"]),
                help="Controls the cadence of generated ticks.",
            )
        )
        seed_text = col_seed.text_input(
            "Random seed (blank = random)",
            value="" if params.get("seed") in (None, "") else str(int(params["seed"])),
            help="Use a fixed seed for reproducible streams or leave blank for randomness.",
        )
        if seed_text.strip():
            try:
                params["seed"] = int(seed_text.strip())
            except ValueError:
                st.warning("Invalid seed input; using random seed instead.")
                params["seed"] = None
        else:
            params["seed"] = None

        col_mu, col_sigma = st.columns(2)
        params["mu"] = float(
            col_mu.number_input(
                "Log-drift (mu)",
                min_value=-0.5,
                max_value=0.5,
                step=0.01,
                value=float(params["mu"]),
            )
        )
        params["sigma"] = float(
            col_sigma.number_input(
                "Diffusion volatility (sigma)",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                value=float(params["sigma"]),
            )
        )

        st.markdown("**Advanced parameters:**")

        col_jump1, col_jump2 = st.columns(2)
        params["jump_lambda"] = float(
            col_jump1.number_input(
                "Jump intensity (lambda)",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                value=float(params["jump_lambda"]),
            )
        )
        params["jump_sigma"] = float(
            col_jump2.number_input(
                "Jump std dev",
                min_value=0.0,
                max_value=0.5,
                step=0.01,
                value=float(params["jump_sigma"]),
            )
        )

        col_jump_mu, col_v0 = st.columns(2)
        params["jump_mu"] = float(
            col_jump_mu.number_input(
                "Jump mean",
                min_value=-0.5,
                max_value=0.5,
                step=0.01,
                value=float(params["jump_mu"]),
            )
        )
        params["v0"] = float(
            col_v0.number_input(
                "Initial volume",
                min_value=1000.0,
                max_value=1_000_000.0,
                step=1000.0,
                value=float(params["v0"]),
            )
        )

        col_phi, col_beta = st.columns(2)
        params["phi"] = float(
            col_phi.slider(
                "Volume persistence (phi)",
                min_value=0.0,
                max_value=0.99,
                step=0.01,
                value=float(params["phi"]),
            )
        )
        params["beta"] = float(
            col_beta.number_input(
                "Volume sensitivity (beta)",
                min_value=0.0,
                max_value=40.0,
                step=1.0,
                value=float(params["beta"]),
            )
        )

        col_gamma, col_sigma_v = st.columns(2)
        params["gamma"] = float(
            col_gamma.number_input(
                "Volume jump bump (gamma)",
                min_value=0.0,
                max_value=5.0,
                step=0.1,
                value=float(params["gamma"]),
            )
        )
        params["sigma_v"] = float(
            col_sigma_v.number_input(
                "Volume noise (sigma_v)",
                min_value=0.0,
                max_value=2.0,
                step=0.05,
                value=float(params["sigma_v"]),
            )
        )

        col_anom_prob, col_anom_min = st.columns(2)
        params["anomaly_prob"] = float(
            col_anom_prob.number_input(
                "Anomaly probability",
                min_value=0.0,
                max_value=0.05,
                step=0.001,
                value=float(params["anomaly_prob"]),
            )
        )
        params["anomaly_min"] = float(
            col_anom_min.number_input(
                "Anomaly min move",
                min_value=0.0,
                max_value=0.5,
                step=0.01,
                value=float(params["anomaly_min"]),
            )
        )

        col_anom_max, col_vol_min = st.columns(2)
        params["anomaly_max"] = float(
            col_anom_max.number_input(
                "Anomaly max move",
                min_value=0.0,
                max_value=0.5,
                step=0.01,
                value=float(params["anomaly_max"]),
            )
        )
        params["vol_anom_min"] = float(
            col_vol_min.number_input(
                "Volume anomaly min multiplier",
                min_value=1.0,
                max_value=20.0,
                step=0.5,
                value=float(params["vol_anom_min"]),
            )
        )

        col_vol_max, col_trading = st.columns(2)
        params["vol_anom_max"] = float(
            col_vol_max.number_input(
                "Volume anomaly max multiplier",
                min_value=1.0,
                max_value=30.0,
                step=0.5,
                value=float(params["vol_anom_max"]),
            )
        )
        params["trading_seconds"] = float(
            col_trading.number_input(
                "Trading day seconds",
                min_value=1000.0,
                max_value=86400.0,
                step=600.0,
                value=float(params["trading_seconds"]),
            )
        )

with right_col:
    wait_placeholder = st.empty()
    chart_placeholder = st.empty()


def _render_ohlcv_list():
    notice_placeholder.empty()
    ticks = _ticks_only()
    if not ticks:
        chart_placeholder.empty()
        with wait_placeholder:
            st.spinner("Waiting for tick data...")
        st.session_state["chart_rendered"] = False
        return

    wait_placeholder.empty()

    df = pd.DataFrame(ticks)[["ts", "price", "volume"]]
    df = df.sort_values("ts")
    df["dt"] = pd.to_datetime(df["ts"], unit="s")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.72, 0.28],
    )
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

    rolling_on = bool(st.session_state.get("rolling_on", False))
    rolling_n = int(st.session_state.get("rolling_n", 20) or 20)

    method = st.session_state.get("anom_method", "Z-Score")
    price_store, vol_store = update_anomaly_store_for_df(
        df,
        method,
        rolling_on=rolling_on,
        rolling_n=rolling_n,
        notice_placeholder=notice_placeholder,
        reset=False,
    )
    price_count = len(price_store)
    vol_count = len(vol_store)

    name_suffix = {
        "Z-Score": "z-score",
        "Abs-Mean+3Std": "abs+3std",
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
            fig.add_vline(x=x_val, line_color=price_color, opacity=0.3, line_width=2.5, row=1, col=1)

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
            fig.add_vline(
                x=x_val,
                line_color=vol_color,
                opacity=0.25,
                line_width=2.0,
                line_dash="dot",
                row=2,
                col=1,
            )

    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title=None,
        yaxis_title="Price",
        yaxis2_title="Volume",
        height=640,
        legend_tracegroupgap=6,
        uirevision="tick-stream",
    )
    fig.update_xaxes(uirevision="tick-stream")
    fig.update_yaxes(uirevision="tick-stream")
    st.session_state["chart_render_count"] = st.session_state.get("chart_render_count", 0) + 1
    chart_key = f"plotly-chart-{st.session_state['chart_render_count']}"
    chart_placeholder.plotly_chart(fig, use_container_width=True, key=chart_key)

    try:
        c1, c2 = metric_placeholder.columns(2)
        c1.metric(label="Price anomalies", value=str(price_count))
        c2.metric(label="Volume anomalies", value=str(vol_count))
    except Exception:
        metric_placeholder.write(f"Price anomalies: {price_count} | Volume anomalies: {vol_count}")


initial_updates = drain_queue()
ticks_len = len(_ticks_only())
runtime_params = st.session_state.get("sim_runtime_params") or st.session_state["sim_params"]
dt_display = float(runtime_params.get("dt", st.session_state["sim_params"].get("dt", 1.0)))
status_placeholder.write(
    f"Simulator: **{st.session_state['sim_status']}** · ticks: {ticks_len} · dt={dt_display:.2f}s"
)
if st.session_state.get("sim_error"):
    sim_error_placeholder.error(st.session_state["sim_error"])
else:
    sim_error_placeholder.empty()

if initial_updates or not st.session_state.get("chart_rendered", False):
    _render_ohlcv_list()
    st.session_state["chart_rendered"] = True
elif not ticks_len:
    _render_ohlcv_list()

if st.session_state["sim_status"] in {"starting", "running", "stopping"}:
    refresh_interval = float(st.session_state.get("refresh_interval", 0.5) or 0.5)
    refresh_interval = max(0.05, refresh_interval)
    max_cycles = int(max(1, 600 / refresh_interval))
    for _ in range(max_cycles):
        changed = drain_queue()
        if changed:
            ticks_len = len(_ticks_only())
            runtime_params = st.session_state.get("sim_runtime_params") or st.session_state["sim_params"]
            dt_display = float(runtime_params.get("dt", st.session_state["sim_params"].get("dt", 1.0)))
            status_placeholder.write(
                f"Simulator: **{st.session_state['sim_status']}** · ticks: {ticks_len} · dt={dt_display:.2f}s"
            )
            if st.session_state.get("sim_error"):
                sim_error_placeholder.error(st.session_state["sim_error"])
            else:
                sim_error_placeholder.empty()
            _render_ohlcv_list()
            st.session_state["chart_rendered"] = True
        time.sleep(refresh_interval)
