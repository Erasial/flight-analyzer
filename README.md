## Flight Data Analyzer

Streamlit app for parsing ArduPilot `.BIN` logs, calculating flight metrics, and visualizing 3D trajectory data.

## Requirements

Dependencies are listed in `requirements.txt`:

- `pandas`
- `numpy`
- `pymavlink`
- `plotly`
- `streamlit`

Install all requirements:

```bash
pip install -r requirements.txt
```

## Run Streamlit

```bash
PYTHONPATH=. streamlit run web/streamlit_app.py
```

## Technologies Used

- `pymavlink`: Reads and parses ArduPilot MAVLink/BIN telemetry logs.
- `pandas`: Stores telemetry as DataFrames and powers data filtering/analysis.
- `numpy`: Handles numeric computations used in trajectory/math utilities.
- `plotly`: Renders the interactive 3D flight trajectory visualization.
- `streamlit`: Provides the web UI for inputs, metrics, tables, and charts.
