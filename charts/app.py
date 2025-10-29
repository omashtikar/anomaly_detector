# app.py
import argparse
import json
import queue
import sys
import threading
import time

import streamlit as st
import pandas as pd

try:
    from ohlcv import ticks_to_ohlcv
except Exception:
    ticks_to_ohlcv = None

try:
    from websocket import WebSocketApp
except ImportError:
    st.error("Missing dependency: install with `pip install websocket-client`")
    st.stop()

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
    def on_open(ws): out_q.put({"_event": "open"})
    def on_message(ws, message):
        try:
            out_q.put(json.loads(message))
        except Exception:
            out_q.put(message)
    def on_error(ws, error): out_q.put({"_event": "error", "error": str(error)})
    def on_close(ws, status_code, close_msg):
        out_q.put({"_event": "close", "status_code": status_code, "reason": close_msg})

    ws = WebSocketApp(url, on_open=on_open, on_message=on_message,
                      on_error=on_error, on_close=on_close)
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
st.caption("Connects to a WebSocket, collects messages, and shows the live count.")

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
        bars_placeholder.info("ohlcv.py missing — cannot compute bars.")
        return
    bars = ticks_to_ohlcv(_ticks_only(), interval_s=10)
    if bars:
        try:
            df = pd.DataFrame(bars)[["start", "end", "open", "high", "low", "close", "volume"]]
            bars_placeholder.dataframe(df, use_container_width=True)
        except Exception:
            # Fallback to plain listing
            bars_placeholder.write(bars)
    else:
        bars_placeholder.info("No OHLCV bars yet — collect more ticks.")

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
