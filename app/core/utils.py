import numpy as np
import pandas as pd

def vectorized_haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Radius in km
    
    # Convert all to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def wgs84_to_enu(df: pd.DataFrame) -> pd.DataFrame:
    if 'Lat' in df.columns and 'Lng' in df.columns and 'Alt' in df.columns:
        lat0 = np.radians(df['Lat'].iloc[0])
        lon0 = np.radians(df['Lng'].iloc[0])
        alt0 = df['Alt'].iloc[0]

        lat = np.radians(df['Lat'].to_numpy())
        lon = np.radians(df['Lng'].to_numpy())
        alt = df['Alt'].to_numpy()

        a = 6378137.0
        e2 = 6.69437999014e-3

        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)

        sin_lat0 = np.sin(lat0)
        cos_lat0 = np.cos(lat0)
        sin_lon0 = np.sin(lon0)
        cos_lon0 = np.cos(lon0)

        n = a / np.sqrt(1 - e2 * sin_lat**2)
        n0 = a / np.sqrt(1 - e2 * sin_lat0**2)

        x = (n + alt) * cos_lat * cos_lon
        y = (n + alt) * cos_lat * sin_lon
        z = (n * (1 - e2) + alt) * sin_lat

        x0 = (n0 + alt0) * cos_lat0 * cos_lon0
        y0 = (n0 + alt0) * cos_lat0 * sin_lon0
        z0 = (n0 * (1 - e2) + alt0) * sin_lat0

        dx = x - x0
        dy = y - y0
        dz = z - z0

        e = -sin_lon0 * dx + cos_lon0 * dy
        n = -sin_lat0 * cos_lon0 * dx - sin_lat0 * sin_lon0 * dy + cos_lat0 * dz
        u = cos_lat0 * cos_lon0 * dx + cos_lat0 * sin_lon0 * dy + sin_lat0 * dz

        enu_df = pd.DataFrame({'Eeast': e, 'North': n, 'Up': u})
        return pd.concat([df.reset_index(drop=True), enu_df], axis=1)
    return df
