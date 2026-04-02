import importlib
import json
from typing import Any

import pandas as pd


BASE_PROMPT_EN = (
    "You are a UAV flight telemetry analysis assistant. "
    "Respond only in English. "
    "Analyze flight metrics and a compact telemetry summary. "
    "Return a practical report in this format:\n"
    "1) Short conclusion (2-4 sentences).\n"
    "2) Key indicators (bullet list).\n"
    "3) Potential risks or anomalies.\n"
    "4) Recommendations for the next flight.\n"
    "Do not invent facts that are not present in the input; if data is insufficient, state that explicitly."
)


class GeminiFlightAssistant:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash") -> None:
        if not api_key:
            raise ValueError("Gemini API key is required")

        try:
            genai = importlib.import_module("google.generativeai")
        except ImportError as exc:
            raise ImportError(
                "google-generativeai is not installed. Install it with: pip install google-generativeai"
            ) from exc

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name=model_name)

    @staticmethod
    def _build_flight_snapshot(df_gps: pd.DataFrame, df_imu: pd.DataFrame) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "gps_samples": int(len(df_gps)),
            "imu_samples": int(len(df_imu)),
            "gps_columns": list(df_gps.columns),
            "imu_columns": list(df_imu.columns),
        }

        if not df_gps.empty:
            if "Alt" in df_gps.columns:
                alt = pd.to_numeric(df_gps["Alt"], errors="coerce").dropna()
                if not alt.empty:
                    snapshot["altitude_start_m"] = float(alt.iloc[0])
                    snapshot["altitude_end_m"] = float(alt.iloc[-1])
                    snapshot["altitude_mean_m"] = float(alt.mean())

            if "Spd" in df_gps.columns:
                spd = pd.to_numeric(df_gps["Spd"], errors="coerce").dropna()
                if not spd.empty:
                    snapshot["speed_mean_m_s"] = float(spd.mean())
                    snapshot["speed_p95_m_s"] = float(spd.quantile(0.95))

            if "VZ" in df_gps.columns:
                vz = pd.to_numeric(df_gps["VZ"], errors="coerce").dropna()
                if not vz.empty:
                    climb = -vz
                    snapshot["climb_rate_mean_m_s"] = float(climb.mean())
                    snapshot["climb_rate_max_m_s"] = float(climb.max())

        return snapshot

    def generate_analysis(
        self,
        metrics: dict[str, float],
        df_gps: pd.DataFrame,
        df_imu: pd.DataFrame,
    ) -> str:
        snapshot = self._build_flight_snapshot(df_gps, df_imu)
        payload = {
            "metrics": metrics,
            "flight_snapshot": snapshot,
        }

        prompt_parts = [
            BASE_PROMPT_EN,
            "Input data (JSON):",
            json.dumps(payload, ensure_ascii=False, indent=2),
        ]

        response = self._model.generate_content("\n\n".join(prompt_parts))
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty response")

        return text
