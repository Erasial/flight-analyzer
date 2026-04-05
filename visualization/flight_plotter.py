import numpy as np
import pandas as pd
import plotly.graph_objects as go
from urllib import error, request
import json


_ELEVATION_CACHE: dict[tuple[int, int], float] = {}


def _coord_cache_key(lat: float, lng: float) -> tuple[int, int]:
	# 1e-5 deg ~= 1.1m at equator; good balance between reuse and terrain fidelity.
	return (int(round(lat * 1e5)), int(round(lng * 1e5)))


def _validate_plot_input(df_gps: pd.DataFrame) -> None:
	required_columns = {"East", "North", "Up"}
	missing = required_columns - set(df_gps.columns)
	if missing:
		raise ValueError(f"df_gps is missing required ENU columns: {sorted(missing)}")

	if "Spd" not in df_gps.columns or "VZ" not in df_gps.columns:
		raise ValueError("df_gps must include 'Spd' and 'VZ' columns for dynamic velocity coloring")


def _coerce_numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
	if column not in df.columns:
		return pd.Series(dtype=float)
	return pd.to_numeric(df[column], errors="coerce")


def _sample_open_elevation(
	lat: np.ndarray,
	lng: np.ndarray,
	timeout_s: float,
) -> np.ndarray | None:
	if len(lat) == 0 or len(lat) != len(lng):
		return None

	keys = [_coord_cache_key(float(la), float(lo)) for la, lo in zip(lat, lng)]
	missing: list[tuple[int, int]] = []
	seen: set[tuple[int, int]] = set()
	for key in keys:
		if key in _ELEVATION_CACHE or key in seen:
			continue
		seen.add(key)
		missing.append(key)

	batch_size = 500
	for start in range(0, len(missing), batch_size):
		end = min(start + batch_size, len(missing))
		batch = missing[start:end]
		locations = [{"latitude": key[0] / 1e5, "longitude": key[1] / 1e5} for key in batch]
		payload = json.dumps({"locations": locations}).encode("utf-8")
		req = request.Request(
			url="https://api.open-elevation.com/api/v1/lookup",
			data=payload,
			headers={"Content-Type": "application/json"},
			method="POST",
		)

		try:
			with request.urlopen(req, timeout=max(0.5, float(timeout_s))) as response:
				body = response.read().decode("utf-8")
		except (error.URLError, TimeoutError, ValueError):
			return None

		try:
			parsed = json.loads(body)
			results = parsed.get("results", [])
			batch_elevations = [float(item.get("elevation")) for item in results]
			if len(batch_elevations) != len(batch):
				return None
			for key, elevation in zip(batch, batch_elevations):
				_ELEVATION_CACHE[key] = elevation
		except (TypeError, ValueError, KeyError, json.JSONDecodeError):
			return None

	values = [_ELEVATION_CACHE.get(key) for key in keys]
	if any(v is None for v in values):
		return None
	return np.asarray(values, dtype=float)


def _enu_square_grid(x: np.ndarray, y: np.ndarray, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
	x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
	y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))

	span = max(x_max - x_min, y_max - y_min, 1.0)
	margin = 0.1 * span
	half = 0.5 * span + margin

	cx = 0.5 * (x_min + x_max)
	cy = 0.5 * (y_min + y_max)

	n = max(15, int(grid_size))
	x_axis = np.linspace(cx - half, cx + half, n)
	y_axis = np.linspace(cy - half, cy + half, n)
	return np.meshgrid(x_axis, y_axis)


def _enu_square_ranges(x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float]]:
	x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
	y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))

	span = max(x_max - x_min, y_max - y_min, 1.0)
	margin = 0.1 * span
	half = 0.5 * span + margin

	cx = 0.5 * (x_min + x_max)
	cy = 0.5 * (y_min + y_max)
	return [cx - half, cx + half], [cy - half, cy + half]


def _enu_to_lat_lng_approx(
	east: np.ndarray,
	north: np.ndarray,
	lat0_deg: float,
	lng0_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
	# Local ENU->geodetic approximation is accurate enough for small flight areas.
	earth_radius_m = 6378137.0
	lat0_rad = np.radians(lat0_deg)
	lat = lat0_deg + np.degrees(north / earth_radius_m)
	lng = lng0_deg + np.degrees(east / (earth_radius_m * np.cos(lat0_rad)))
	return lat, lng


def _prepare_ground_surface(
	df_gps: pd.DataFrame,
	x: np.ndarray,
	y: np.ndarray,
	ground_grid_size: int,
	open_elevation_timeout_s: float,
	terrain_altitude_origin: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	lat = _coerce_numeric_column(df_gps, "Lat")
	lng = _coerce_numeric_column(df_gps, "Lng")
	alt = _coerce_numeric_column(df_gps, "Alt")
	valid_geo = lat.notna() & lng.notna() & alt.notna()
	if valid_geo.sum() < 2:
		raise ValueError("Open-Elevation terrain requires valid 'Lat', 'Lng', and 'Alt' in GPS data")

	lat0 = float(lat.loc[valid_geo].iloc[0])
	lng0 = float(lng.loc[valid_geo].iloc[0])
	if terrain_altitude_origin is None:
		alt0 = float(alt.loc[valid_geo].iloc[0])
	else:
		alt0 = float(terrain_altitude_origin)

	gx, gy = _enu_square_grid(x, y, grid_size=ground_grid_size)
	grid_lat, grid_lng = _enu_to_lat_lng_approx(gx, gy, lat0_deg=lat0, lng0_deg=lng0)

	elevation_asl = _sample_open_elevation(
		grid_lat.ravel(),
		grid_lng.ravel(),
		timeout_s=open_elevation_timeout_s,
	)
	if elevation_asl is None:
		raise ValueError("Failed to fetch terrain from Open-Elevation API")

	gz = elevation_asl.reshape(gx.shape) - alt0
	return gx, gy, gz


def _build_trajectory(df_gps: pd.DataFrame) -> pd.DataFrame:
	e = pd.to_numeric(df_gps["East"], errors="coerce")
	n = pd.to_numeric(df_gps["North"], errors="coerce")
	u = pd.to_numeric(df_gps["Up"], errors="coerce")
	spd = pd.to_numeric(df_gps["Spd"], errors="coerce")
	vz = pd.to_numeric(df_gps["VZ"], errors="coerce")
	climb = -vz  # ArduPilot convention: negative VZ means climbing.

	trajectory = pd.DataFrame({"East": e, "North": n, "Up": u, "Spd": spd, "VZ": vz, "ClimbRate": climb})
	for optional_col in ["Lat", "Lng", "Alt"]:
		if optional_col in df_gps.columns:
			trajectory[optional_col] = pd.to_numeric(df_gps[optional_col], errors="coerce")
	if "TimeUS" in df_gps.columns:
		trajectory["TimeUS"] = pd.to_numeric(df_gps["TimeUS"], errors="coerce")

	trajectory = trajectory.dropna(subset=["East", "North", "Up", "Spd", "VZ", "ClimbRate"])
	if trajectory.empty or len(trajectory) < 2:
		raise ValueError("df_gps has no valid ENU points to plot")

	return trajectory


def _resolve_speed_unit(speed_unit: str) -> tuple[float, str]:
	unit = speed_unit.strip().lower()
	if unit in {"km/h", "км/год"}:
		return 3.6, "км/год"
	return 1.0, "м/с"


def _resolve_color_metric(
	trajectory: pd.DataFrame,
	color_by: str,
	speed_unit: str,
) -> tuple[np.ndarray, str, str, np.ndarray, np.ndarray, str, str]:
	color_mode = color_by.strip().lower()
	unit_factor, speed_unit_label = _resolve_speed_unit(speed_unit)
	ground_speed = trajectory["Spd"].to_numpy(dtype=float) * unit_factor
	climb_rate = trajectory["ClimbRate"].to_numpy(dtype=float) * unit_factor

	if color_mode == "ground":
		return (
			ground_speed,
			f"Ground Speed ({speed_unit_label})",
			"Ground Speed",
			ground_speed,
			climb_rate,
			speed_unit_label,
			speed_unit_label,
		)

	if color_mode == "vertical":
		return (
			climb_rate,
			f"Climb Rate ({speed_unit_label})",
			"Climb Rate",
			ground_speed,
			climb_rate,
			speed_unit_label,
			speed_unit_label,
		)

	if color_mode == "time":
		if "TimeUS" in trajectory.columns and not trajectory["TimeUS"].isna().all():
			time_us = trajectory["TimeUS"].to_numpy(dtype=float)
			time_color = (time_us - float(time_us[0])) / 1e6
		else:
			time_color = np.arange(len(trajectory), dtype=float)
		return (
			time_color,
			"Flight Time (s)",
			"Time",
			ground_speed,
			climb_rate,
			speed_unit_label,
			"s",
		)

	velocity_color = np.sqrt(ground_speed**2 + climb_rate**2)
	return (
		velocity_color,
		f"Total Speed ({speed_unit_label})",
		"Total Speed",
		ground_speed,
		climb_rate,
		speed_unit_label,
		speed_unit_label,
	)


def _build_figure(
	x: np.ndarray,
	y: np.ndarray,
	z: np.ndarray,
	ground_speed: np.ndarray,
	climb_rate: np.ndarray,
	velocity_color: np.ndarray,
	color_title: str,
	plot_title_suffix: str,
	speed_unit_label: str,
	color_metric_unit_label: str,
	show_ground: bool,
	ground_surface: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
	x_axis_range: list[float],
	y_axis_range: list[float],
):
	fig = go.Figure()

	x_span = float(x_axis_range[1] - x_axis_range[0])
	y_span = float(y_axis_range[1] - y_axis_range[0])
	xy_span = max(x_span, y_span, 1.0)
	z_span = max(1e-6, float(np.nanmax(z) - np.nanmin(z)))
	# Keep XY square for map fidelity, but boost Z readability to avoid flattened perception.
	z_aspect = float(np.clip((z_span / xy_span) * 2.4, 0.95, 1.8))

	if show_ground and ground_surface is not None:
		gx, gy, gz = ground_surface
		fig.add_trace(
			go.Surface(
				x=gx,
				y=gy,
				z=gz,
				name="Ground Surface",
				opacity=0.55,
				showscale=False,
				colorscale=[[0.0, "#6E8B3D"], [0.5, "#A78A5A"], [1.0, "#D8C4A3"]],
				hovertemplate="Ground<br>E: %{x:.2f} m<br>N: %{y:.2f} m<br>U: %{z:.2f} m<extra></extra>",
			)
		)

	fig.add_trace(
		go.Scatter3d(
			x=x,
			y=y,
			z=z,
			mode="lines+markers",
			name="Trajectory",
			line={
				"color": velocity_color,
				"colorscale": "Turbo",
				"width": 6,
				"colorbar": {
					"title": {"text": color_title, "side": "right"},
					"thickness": 12,
					"len": 0.55,
					"x": 0.92,
					"y": 0.5,
					"yanchor": "middle",
					"xpad": 6,
				},
				"cmin": float(np.nanmin(velocity_color)),
				"cmax": float(np.nanmax(velocity_color)),
			},
			marker={
				"size": 2,
				"color": velocity_color,
				"colorscale": "Turbo",
				"showscale": False,
			},
			customdata=np.column_stack((ground_speed, climb_rate, velocity_color)),
			hovertemplate=(
				"E: %{x:.2f} m<br>"
				"N: %{y:.2f} m<br>"
				"U: %{z:.2f} m<br>"
				f"Ground Speed: %{{customdata[0]:.2f}} {speed_unit_label}<br>"
				f"Climb Rate: %{{customdata[1]:.2f}} {speed_unit_label}<br>"
				f"Color Metric: %{{customdata[2]:.2f}} {color_metric_unit_label}<extra></extra>"
			),
		)
	)

	fig.add_trace(
		go.Scatter3d(
			x=[x[0]],
			y=[y[0]],
			z=[z[0]],
			mode="markers",
			name="Start",
			marker={"size": 6, "color": "green"},
		)
	)
	fig.add_trace(
		go.Scatter3d(
			x=[x[-1]],
			y=[y[-1]],
			z=[z[-1]],
			mode="markers",
			name="End",
			marker={"size": 6, "color": "red"},
		)
	)

	fig.update_layout(
		title=f"Flight Trajectory (ENU) - Colored by {plot_title_suffix}",
		legend={"x": 0.01, "y": 0.99, "yanchor": "top", "bgcolor": "rgba(255,255,255,0.7)"},
		scene={
			"xaxis": {"title": "East (m)", "range": x_axis_range, "autorange": False},
			"yaxis": {"title": "North (m)", "range": y_axis_range, "autorange": False},
			"zaxis": {"title": "Up (m)"},
			"aspectmode": "manual",
			"aspectratio": {"x": 1.0, "y": 1.0, "z": z_aspect},
		},
		margin={"l": 0, "r": 24, "b": 0, "t": 40},
	)

	return fig


def plot_flight_path_3d(
	df_gps: pd.DataFrame,
	output_html: str = "flight_trajectory_enu.html",
	auto_open: bool = False,
	color_by: str = "combined",
	speed_unit: str = "m/s",
	show_ground: bool = True,
	ground_grid_size: int = 20,
	open_elevation_timeout_s: float = 4.0,
	terrain_altitude_origin: float | None = None,
):
	"""Build an interactive Plotly 3D trajectory with optional Open-Elevation terrain."""
	_validate_plot_input(df_gps)
	trajectory = _build_trajectory(df_gps)

	(
		velocity_color,
		color_title,
		plot_title_suffix,
		ground_speed,
		climb_rate,
		speed_unit_label,
		color_metric_unit_label,
	) = _resolve_color_metric(trajectory, color_by, speed_unit)

	x = np.ravel(trajectory["East"].to_numpy(dtype=float))
	y = np.ravel(trajectory["North"].to_numpy(dtype=float))
	z = np.ravel(trajectory["Up"].to_numpy(dtype=float))
	ground_speed = np.ravel(ground_speed)
	climb_rate = np.ravel(climb_rate)
	x_axis_range, y_axis_range = _enu_square_ranges(x, y)

	ground_surface = None
	if show_ground:
		ground_surface = _prepare_ground_surface(
			df_gps=trajectory,
			x=x,
			y=y,
			ground_grid_size=ground_grid_size,
			open_elevation_timeout_s=open_elevation_timeout_s,
			terrain_altitude_origin=terrain_altitude_origin,
		)

	fig = _build_figure(
		x,
		y,
		z,
		ground_speed,
		climb_rate,
		velocity_color,
		color_title,
		plot_title_suffix,
		speed_unit_label,
		color_metric_unit_label,
		show_ground=show_ground,
		ground_surface=ground_surface,
		x_axis_range=x_axis_range,
		y_axis_range=y_axis_range,
	)

	if output_html:
		fig.write_html(output_html, include_plotlyjs="cdn", auto_open=auto_open)
		print(f"Saved interactive trajectory plot to {output_html}")

	return fig
