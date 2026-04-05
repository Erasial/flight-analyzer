import numpy as np
import pandas as pd


def _rotation_body_to_earth(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return ZYX body->earth rotation matrix from Euler angles (radians)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def cumulative_trapezoidal(time_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Pure cumulative trapezoidal integration for 1D arrays."""
    t = np.asarray(time_s, dtype=float)
    v = np.asarray(values, dtype=float)
    if t.ndim != 1 or v.ndim != 1 or len(t) != len(v):
        raise ValueError("time_s and values must be 1D arrays with equal length")
    if len(t) == 0:
        return np.array([], dtype=float)

    dt = np.diff(t)
    increments = 0.5 * (v[1:] + v[:-1]) * dt
    return np.concatenate(([0.0], np.cumsum(increments)))


def integrate_velocity_from_imu_trapezoidal(
    time_s: np.ndarray,
    acc_body_xyz: np.ndarray,
    gyro_body_xyz: np.ndarray,
    gravity_mps2: float = 9.80665,
) -> dict[str, np.ndarray]:
    """Integrate IMU acceleration to velocity in earth frame using gyro orientation updates.

    Inputs are pure arrays: time (s), body acceleration (m/s^2), body gyro rates (rad/s).
    """
    t = np.asarray(time_s, dtype=float)
    acc = np.asarray(acc_body_xyz, dtype=float)
    gyro = np.asarray(gyro_body_xyz, dtype=float)

    if t.ndim != 1:
        raise ValueError("time_s must be a 1D array")
    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError("acc_body_xyz must be shape (N, 3)")
    if gyro.ndim != 2 or gyro.shape[1] != 3:
        raise ValueError("gyro_body_xyz must be shape (N, 3)")
    if len(t) != len(acc) or len(t) != len(gyro):
        raise ValueError("time, acceleration, and gyro arrays must have matching lengths")
    if len(t) == 0:
        return {
            "acc_earth_x": np.array([], dtype=float),
            "acc_earth_y": np.array([], dtype=float),
            "acc_earth_z": np.array([], dtype=float),
            "vel_earth_x": np.array([], dtype=float),
            "vel_earth_y": np.array([], dtype=float),
            "vel_earth_z": np.array([], dtype=float),
            "vel_norm": np.array([], dtype=float),
        }

    roll = 0.0
    pitch = 0.0
    yaw = 0.0
    acc_earth = np.zeros_like(acc, dtype=float)

    for i in range(len(t)):
        if i > 0:
            dt = max(0.0, t[i] - t[i - 1])
            roll += gyro[i - 1, 0] * dt
            pitch += gyro[i - 1, 1] * dt
            yaw += gyro[i - 1, 2] * dt

        rotation = _rotation_body_to_earth(roll, pitch, yaw)
        acc_earth[i, :] = rotation @ acc[i, :]

    # Remove gravity on earth-frame vertical axis.
    acc_earth[:, 2] = acc_earth[:, 2] - float(gravity_mps2)

    vel_x = cumulative_trapezoidal(t, acc_earth[:, 0])
    vel_y = cumulative_trapezoidal(t, acc_earth[:, 1])
    vel_z = cumulative_trapezoidal(t, acc_earth[:, 2])
    vel_norm = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)

    return {
        "acc_earth_x": acc_earth[:, 0],
        "acc_earth_y": acc_earth[:, 1],
        "acc_earth_z": acc_earth[:, 2],
        "vel_earth_x": vel_x,
        "vel_earth_y": vel_y,
        "vel_earth_z": vel_z,
        "vel_norm": vel_norm,
    }

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

        enu_df = pd.DataFrame({'East': e, 'North': n, 'Up': u})
        return pd.concat([df.reset_index(drop=True), enu_df], axis=1)
    return df
