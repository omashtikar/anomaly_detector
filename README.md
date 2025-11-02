# Trading212 Anomaly Detector

An interactive Streamlit dashboard for experimenting with real-time market anomaly detection. The app ships with a realistic jump–diffusion tick simulator, multiple anomaly detection strategies, live visualisation of price/volume streams, and a set of controls for tailoring both the feed and the analytics. Everything runs locally – no external data or network calls are required.

---

## Key Features

- **Integrated tick simulator** – Generates streaming price/volume data using a Merton jump–diffusion model with configurable drift, volatility, jumps, AR(1) volume dynamics, and random anomaly shocks.
- **Four detection strategies out of the box** – Z-Score, Abs-Mean+3Std, Isolation Forest, and Prophet residuals, each with an inline explanation beneath the chart.
- **Live Plotly dashboard** – Price line, volume bars, and anomaly markers with colour-coded vertical emphasis, plus counters for price/volume hits.
- **Dynamic analytics controls** – Toggle rolling windows, adjust screen refresh cadence, and switch detection methods without restarting the app.
- **Simulator tuning panel** – Change tick cadence, random seed, jump intensity, anomaly probabilities, volume dynamics, and more via a single collapsible section.
- **Embeddable generator** – Use `exchange_server/sim_feed.py` as a standalone CLI or WebSocket service to drive external consumers.
- **Asset pack for documentation** – Pre-rendered screenshots in `assets/` to support presentations or onboarding material.

---

## Repository Layout

| Path | Description |
| ---- | ----------- |
| `charts/app.py` | Streamlit application combining simulator controls, anomaly detection, and visualisation. |
| `charts/utils.py` | Shared anomaly utilities (rate-of-change helpers, z-score and absolute-threshold detectors). |
| `charts/ohlcv.py` | Legacy chart helpers (currently unused but retained for reference). |
| `exchange_server/sim_feed.py` | Jump–diffusion simulator with CLI and WebSocket modes. |
| `assets/` | Reference screenshots for documentation and demos. |
| `requirements.txt` | Locked dependency versions for the Streamlit app and simulator. |

---

## Architecture Overview

1. **Simulation layer**  
   Implemented in `exchange_server/sim_feed.py`. The `jd_price_volume_stream` generator produces tick dictionaries with timestamp, price, volume, return, jump flags, and anomaly metadata. Parameters are bundled in the `JDParams` dataclass.

2. **Analytics layer**  
   Functions in `charts/utils.py` provide reusable anomaly logic (z-score and absolute-threshold detection). Additional methods (Isolation Forest, Prophet) are loaded lazily inside the app to minimise dependencies when not needed.

3. **Presentation layer**  
   `charts/app.py` (Streamlit) orchestrates:
   - Session state for the simulator, anomaly history, and UI components.
   - Background worker threads to keep the simulator responsive.
   - Plotly visualisations and explanatory text for the chosen method.

---

## Anomaly Detection Methods

| Method | Description | Dependencies |
| ------ | ----------- | ------------ |
| **Z-Score** | Flags ticks where the standardised price/volume change exceeds three standard deviations. | `numpy` + custom logic |
| **Abs-Mean+3Std** | Uses absolute tick changes versus mean + 3σ, highlighting large moves irrespective of direction. | `numpy` + custom logic |
| **Isolation Forest** | Runs an unsupervised Isolation Forest on normalised price and volume to isolate outliers in feature space. | `scikit-learn` |
| **Model (Prophet)** | Fits a Prophet time-series model and marks residuals > mean + 3σ of absolute errors. | `prophet` (or `fbprophet`) |

The active method’s summary is rendered below the chart for quick onboarding.

---

## Simulator Controls

All simulator knobs live inside the “Simulator parameters” expander on the left pane:

- **Tick cadence** (`dt`) and reproducible **seed**.
- **Price dynamics** – drift, volatility, jump intensity/mean/std.
- **Volume dynamics** – AR(1) persistence, sensitivity to returns, jump amplification, noise.
- **Anomaly shocks** – per-tick probability, min/max price shock, volume multipliers.
- **Clock** – total “trading seconds” used to loop the intraday U-shape profile.

The app also exposes:

- **Rolling window toggle** – Restrict detection to the latest N ticks (20/50/100).
- **Screen refresh rate** – Control how often the chart re-renders while streaming.
- **Anomaly counters** – Real-time price/volume anomaly totals per method.

---

## Quick Start

1. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate                # Linux/macOS
   # or
   venv\Scripts\activate                   # Windows
   ```

2. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit dashboard**

   ```bash
   streamlit run charts/app.py
   ```

   The app opens in your browser (default: http://localhost:8501). Use the left panel to choose detection methods and start the simulator.

---

## Running the Simulator Standalone

The simulator can be reused outside Streamlit:

```bash
python exchange_server/sim_feed.py --mode stdout --dt 0.5 --anomaly-prob 0.01
```

Outputs newline-delimited JSON ticks to stdout.

```bash
python exchange_server/sim_feed.py --mode websocket --host 127.0.0.1 --port 8765
```

Serves ticks over WebSocket (`ws://127.0.0.1:8765/prices`). Any WebSocket client can subscribe and consume the stream.

---

## Using the Dashboard

1. Pick an anomaly detection method. The app stops the simulator, recomputes historical anomalies for that method, and shows its description under the chart.
2. Press **Start simulator** to begin streaming. Price and volume update live, anomalies are pinned as coloured markers, and vertical lines highlight their timestamps.
3. Toggle a rolling window or tweak screen refresh rate to explore different sensitivities.
4. Expand **Simulator parameters** to experiment with regime shifts, jumps, or anomaly frequency.
5. Stop the simulator at any time to inspect the static historical view.

---

## Extending the Project

- **Add a new anomaly method**  
  Implement the logic (e.g. in `charts/utils.py` or inline), register it in `ANOM_METHODS`, add a description, and update `update_anomaly_store_for_df` in `charts/app.py`.

- **Stream external data**  
  Replace `jd_price_volume_stream` in `charts/app.py` with your own producer that writes dictionaries matching the expected format (`ts`, `price`, `volume`).

- **Persist anomalies**  
  Use the `st.session_state["anoms"]` dictionary (per-method price/volume stores) as a basis for exporting or alerting.

---

## Troubleshooting

- **Duplicate element errors** – Streamlit caches keys aggressively. The app already assigns unique keys per render; if you add new charts, ensure their keys change per update.
- **Isolation Forest / Prophet unavailable** – Install the optional dependencies listed in `requirements.txt`. The app will display helper messages if a library is missing.
- **Busy refresh loop** – Increase the “Screen refresh rate” slider if your CPU spikes; decrease it for more granular updates.
- **No anomalies** – Increase `anomaly_prob`, widen the rolling window, or choose a more sensitive method (Abs-Mean+3Std).

---

Enjoy exploring anomaly detection scenarios without needing a live market feed!
