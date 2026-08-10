import os
import tempfile
import uuid
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.parsers.base import ParseStatus
from app.parsers.binary import BinaryDataParser
from app.services.analyzer import AnalysisService
from app.services.data_quality import assess_metric_quality
from app.services.pipeline import collect_metrics, prepare_telemetry_frames

app = FastAPI(title="Flight Data Analyzer API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances and storage
parser = BinaryDataParser()
analyzer = AnalysisService()
# In-memory storage for analysis results (using UUID as key)
results_storage: dict[str, Any] = {}

@app.post("/analyze")
async def analyze_flight_log(file: Annotated[UploadFile, File(...)]):
    """
    Upload a .BIN file, process it, and return a unique identifier for the result.
    """
    if not file.filename or not file.filename.upper().endswith('.BIN'):
        raise HTTPException(status_code=400, detail="Only .BIN files are allowed.")

    temp_path = None
    try:
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".BIN") as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name

        # Parse the binary file into dataframes
        parse_result = parser.parse_with_diagnostics(temp_path)
        if parse_result.status is ParseStatus.REJECTED:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": parse_result.error or "BIN file could not be decoded",
                    "integrity_status": parse_result.status.value,
                    "warnings": list(parse_result.warnings),
                },
            )

        # Prepare telemetry frames (GPS and IMU)
        telemetry = prepare_telemetry_frames(analyzer, parse_result.dataframes)

        if telemetry.df_gps.empty:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "GPS data is missing or empty in this log file.",
                    "integrity_status": parse_result.status.value,
                    "warnings": list(parse_result.warnings),
                },
            )
        
        # Collect various flight metrics
        metrics = collect_metrics(analyzer, telemetry.df_gps, telemetry.df_imu)
        metric_quality = assess_metric_quality(metrics, telemetry.quality_report)

        # Merge ATT and GPS data for synchronized telemetry
        if not telemetry.df_att.empty and not telemetry.df_gps.empty:
            # ArduPilot data usually has TimeUS for all messages. 
            # We use merge_asof for nearest-time synchronization.
            df_gps_sorted = telemetry.df_gps.sort_values('TimeUS')
            df_att_sorted = telemetry.df_att.sort_values('TimeUS')
            
            # Ensure TimeUS is numeric for merging
            df_gps_sorted['TimeUS'] = pd.to_numeric(df_gps_sorted['TimeUS'])
            df_att_sorted['TimeUS'] = pd.to_numeric(df_att_sorted['TimeUS'])
            
            df_combined = pd.merge_asof(
                df_gps_sorted,
                df_att_sorted,
                on='TimeUS',
                direction='nearest'
            )
        else:
            df_combined = telemetry.df_gps

        # Get top 100 rows for preview
        table_data = df_combined.head(100).to_dict(orient="records")

        # Prepare chart points including raw GPS for mapping, ENU for 3D, and Attitude
        available_cols = df_combined.columns.tolist()
        # Ensure we pick Lat, Lng, Alt, East, North, Up as well
        target_cols = [
            'Lat', 'Lng', 'Alt', 'East', 'North', 'Up', 'TimeUS',
            'Roll', 'Pitch', 'Yaw', 'Yaw_y',
        ]
        cols_to_use = [c for c in target_cols if c in available_cols]
        
        # Explicitly make a copy of the slice
        chart_data = df_combined[cols_to_use].copy()
        # Rename Yaw_y to Yaw if it exists
        if 'Yaw_y' in chart_data.columns:
            chart_data = chart_data.rename(columns={'Yaw_y': 'Yaw'})
        if not chart_data.empty:
            start_time = chart_data['TimeUS'].iloc[0]
            chart_data['RelativeTime'] = (chart_data['TimeUS'] - start_time) / 1e6
        
        chart_points = chart_data.to_dict(orient="records")

        # Store the result with a unique ID
        result_id = str(uuid.uuid4())
        results_storage[result_id] = {
            "filename": file.filename,
            "integrity": {
                "status": parse_result.status.value,
                "warnings": list(parse_result.warnings),
                "decoded_message_count": parse_result.decoded_message_count,
                "decoder_diagnostic_count": parse_result.diagnostics.total_lines,
                "suppressed_diagnostic_count": parse_result.diagnostics.suppressed_lines,
                "artifact_size_bytes": parse_result.artifact_size_bytes,
                "artifact_sha256": parse_result.artifact_sha256,
            },
            "data_quality": telemetry.quality_report.to_dict(),
            "flight_events": telemetry.event_report.to_dict(),
            "metrics": metrics,
            "metric_quality": {
                name: assessment.to_dict()
                for name, assessment in metric_quality.items()
            },
            "table_preview": table_data,
            "chart_points": chart_points
        }

        return {"result_id": result_id}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/results/{result_id}")
async def get_analysis_result(result_id: str):
    """
    Retrieve stored flight analysis data by result_id.
    """
    if result_id not in results_storage:
        raise HTTPException(status_code=404, detail="Result ID not found.")
    
    return results_storage[result_id]

@app.get("/")
async def root():
    return {"message": "Flight Data Analyzer API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
