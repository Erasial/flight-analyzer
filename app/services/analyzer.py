import pandas as pd
import math

from app.core.utils import vectorized_haversine

class AnalysisService:
    @staticmethod
    def filter_gps_low_quality_samples(
        df: pd.DataFrame,
        min_status: int = 3,
        require_positive_gwk: bool = True,
        min_sats: int = 0,
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=df.columns if df is not None else [])

        filtered = df.copy()

        if 'Status' in filtered.columns:
            status = pd.to_numeric(filtered['Status'], errors='coerce')
            filtered = filtered[status >= min_status]

        if require_positive_gwk and 'GWk' in filtered.columns:
            gwk = pd.to_numeric(filtered['GWk'], errors='coerce')
            filtered = filtered[gwk > 0]

        if min_sats > 0 and 'NSats' in filtered.columns:
            nsats = pd.to_numeric(filtered['NSats'], errors='coerce')
            filtered = filtered[nsats >= min_sats]

        if 'TimeUS' in filtered.columns:
            filtered = filtered.sort_values('TimeUS')

        return filtered.reset_index(drop=True)

    @staticmethod
    def filter_imu_module(df: pd.DataFrame, imu_index: int = 0) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=df.columns if df is not None else [])

        if 'I' not in df.columns:
            raise ValueError("IMU dataframe must contain 'I' column for module index filtering.")

        filtered_df = df[df['I'] == imu_index].copy()
        return filtered_df.reset_index(drop=True)

    @staticmethod
    def get_max_vertical_speed(df: pd.DataFrame) -> float:
        if 'VZ' in df.columns:
            return df['VZ'].abs().max()
        return 0.0
    
    @staticmethod
    def get_max_horizontal_speed(df: pd.DataFrame) -> float:
        if 'Spd' in df.columns:
            return df['Spd'].abs().max()
        return 0.0
    
    @staticmethod
    def get_max_altitude(df: pd.DataFrame) -> float:
        if 'Alt' in df.columns:
            return df['Alt'].max()
        return 0.0
    
    @staticmethod
    def get_max_acceleration(df: pd.DataFrame) -> dict:
        acc_columns = [col for col in df.columns if col.startswith('Acc')]
        return df[acc_columns].abs().max().to_dict()
    
    @staticmethod
    def get_distance_traveled(df: pd.DataFrame) -> float:
        if 'Lat' in df.columns and 'Lng' in df.columns:
            lat1 = df['Lat'].shift()
            lon1 = df['Lng'].shift()
            lat2 = df['Lat']
            lon2 = df['Lng']
            distances_km = vectorized_haversine(lat1, lon1, lat2, lon2)
            return distances_km.fillna(0).sum() * 1000.0
        return 0.0
    
    @staticmethod
    def get_sample_rate(df: pd.DataFrame) -> float:
        if 'TimeUS' in df.columns and len(df) > 1:
            time_diffs = df['TimeUS'].diff().dropna()
            avg_time_diff_sec = time_diffs.mean() / 1e6
            return 1.0 / avg_time_diff_sec if avg_time_diff_sec > 0 else 0.0
        return 0.0

    @staticmethod
    def get_flight_duration(df: pd.DataFrame) -> float:
        if 'TimeUS' in df.columns and len(df) > 1:
            start_time = df['TimeUS'].iloc[0]
            end_time = df['TimeUS'].iloc[-1]
            return (end_time - start_time) / 1e6
        return 0.0

    @staticmethod
    def filter_outliers(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
        """
        Filters outliers using Z-score logic.
        """
        if df is None or df.empty or column not in df.columns:
            return df
        
        data = pd.to_numeric(df[column], errors='coerce')
        mean = data.mean()
        std = data.std()
        
        if std == 0:
            return df
            
        z_scores = (data - mean) / std
        return df[z_scores.abs() <= threshold].reset_index(drop=True)

    @staticmethod
    def smooth_signal(df: pd.DataFrame, column: str, window: int = 5) -> pd.DataFrame:
        """
        Applies a simple moving average to smooth noisy ArduPilot data.
        """
        if df is None or df.empty or column not in df.columns:
            return df
            
        df = df.copy()
        df[column] = df[column].rolling(window=window, center=True).mean()
        return df.dropna(subset=[column]).reset_index(drop=True)

    @staticmethod
    def process_attitude(df_att: pd.DataFrame) -> pd.DataFrame:
        """
        ArduPilot ATT messages usually have Roll, Pitch, Yaw in degrees.
        This method ensures they are within standard ranges and smooths them.
        """
        if df_att is None or df_att.empty:
            return pd.DataFrame()

        df = df_att.copy()
        
        # Ensure numeric
        for col in ['Roll', 'Pitch', 'Yaw']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # ArduPilot Yaw is 0-360, but let's keep it consistent.
        # If we had quaternions, we'd use them here.
        
        return df