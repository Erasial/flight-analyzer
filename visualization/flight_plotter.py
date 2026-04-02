import numpy as np
import pandas as pd


def _load_plotly():
	try:
		import plotly.graph_objects as go
	except ImportError as exc:
		raise ImportError(
			"plotly is required for interactive plotting. Install it with: pip install plotly"
		) from exc
	return go


def _extract_axis(df: pd.DataFrame, column_name: str) -> pd.Series:
    """Return a 1D numeric series even when duplicate column names exist."""
    axis_data = df[column_name]
    if isinstance(axis_data, pd.DataFrame):
        axis_data = axis_data.iloc[:, -1]

    axis_series = pd.to_numeric(axis_data, errors="coerce")
    return axis_series


def plot_flight_path_3d(
	df_gps: pd.DataFrame,
	output_html: str = "flight_trajectory_enu.html",
	auto_open: bool = False,
	color_by: str = "combined",
):
	"""Build an interactive Plotly 3D trajectory with velocity-based dynamic coloring."""
	required_columns = {"Eeast", "North", "Up"}
	missing = required_columns - set(df_gps.columns)
	if missing:
		raise ValueError(f"df_gps is missing required ENU columns: {sorted(missing)}")

	if "Spd" not in df_gps.columns or "VZ" not in df_gps.columns:
		raise ValueError("df_gps must include 'Spd' and 'VZ' columns for dynamic velocity coloring")

	e = _extract_axis(df_gps, "Eeast")
	n = _extract_axis(df_gps, "North")
	u = _extract_axis(df_gps, "Up")
	spd = _extract_axis(df_gps, "Spd")
	vz = _extract_axis(df_gps, "VZ")
	climb = -vz  # ArduPilot convention: negative VZ means climbing.

	trajectory = pd.DataFrame(
		{"Eeast": e, "North": n, "Up": u, "Spd": spd, "VZ": vz, "ClimbRate": climb}
	).dropna()
	if trajectory.empty or len(trajectory) < 2:
		raise ValueError("df_gps has no valid ENU points to plot")

	color_mode = color_by.strip().lower()
	if color_mode == "ground":
		velocity_color = trajectory["Spd"].to_numpy(dtype=float)
		color_title = "Ground Speed (m/s)"
		plot_title_suffix = "Ground Speed"
	elif color_mode == "vertical":
		velocity_color = trajectory["ClimbRate"].to_numpy(dtype=float)
		color_title = "Climb Rate (m/s)"
		plot_title_suffix = "Climb Rate"
	else:
		velocity_color = np.sqrt(
			trajectory["Spd"].to_numpy(dtype=float) ** 2 + trajectory["ClimbRate"].to_numpy(dtype=float) ** 2
		)
		color_title = "Total Speed (m/s)"
		plot_title_suffix = "Total Speed"

	x = np.ravel(trajectory["Eeast"].to_numpy(dtype=float))
	y = np.ravel(trajectory["North"].to_numpy(dtype=float))
	z = np.ravel(trajectory["Up"].to_numpy(dtype=float))
	ground_speed = np.ravel(trajectory["Spd"].to_numpy(dtype=float))
	climb_rate = np.ravel(trajectory["ClimbRate"].to_numpy(dtype=float))

	go = _load_plotly()

	fig = go.Figure()
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
				"Ground Speed: %{customdata[0]:.2f} m/s<br>"
				"Climb Rate: %{customdata[1]:.2f} m/s<br>"
				"Color Metric: %{customdata[2]:.2f} m/s<extra></extra>"
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
			"xaxis_title": "East (m)",
			"yaxis_title": "North (m)",
			"zaxis_title": "Up (m)",
		},
		margin={"l": 0, "r": 24, "b": 0, "t": 40},
	)

	if output_html:
		fig.write_html(output_html, include_plotlyjs="cdn", auto_open=auto_open)
		print(f"Saved interactive trajectory plot to {output_html}")

	return fig
