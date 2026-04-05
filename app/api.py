import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import os
import tempfile
from typing import List, Dict, Any

from app.parsers.binary import BinaryDataParser
from app.services.analyzer import AnalysisService
from app.services.pipeline import prepare_telemetry_frames, collect_metrics

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
results_storage: Dict[str, Any] = {}

@app.post("/analyze")
async def analyze_flight_log(file: UploadFile = File(...)):
    """
    Upload a .BIN file, process it, and return a unique identifier for the result.
    """
    if not file.filename.endswith('.BIN'):
        raise HTTPException(status_code=400, detail="Only .BIN files are allowed.")

    temp_path = None
    try:
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".BIN") as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name

        # Parse the binary file into dataframes
        dataframes = parser.parse(temp_path)

        # Prepare telemetry frames (GPS and IMU)
        telemetry = prepare_telemetry_frames(analyzer, dataframes)

        if telemetry.df_gps.empty:
            raise HTTPException(status_code=400, detail="GPS data is missing or empty in this log file.")
        
        # Collect various flight metrics
        metrics = collect_metrics(analyzer, telemetry.df_gps, telemetry.df_imu)

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
        target_cols = ['Lat', 'Lng', 'Alt', 'East', 'North', 'Up', 'TimeUS', 'Roll', 'Pitch', 'Yaw', 'Yaw_y']
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
            "metrics": metrics,
            "table_preview": table_data,
            "chart_points": chart_points
        }

        return {"result_id": result_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
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
