import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _validate_plot_input(df_gps: pd.DataFrame) -> None:
	required_columns = {"Eeast", "North", "Up"}
	missing = required_columns - set(df_gps.columns)
	if missing:
		raise ValueError(f"df_gps is missing required ENU columns: {sorted(missing)}")

	if "Spd" not in df_gps.columns or "VZ" not in df_gps.columns:
		raise ValueError("df_gps must include 'Spd' and 'VZ' columns for dynamic velocity coloring")


def _build_trajectory(df_gps: pd.DataFrame) -> pd.DataFrame:
	e = pd.to_numeric(df_gps["Eeast"], errors="coerce")
	n = pd.to_numeric(df_gps["North"], errors="coerce")
	u = pd.to_numeric(df_gps["Up"], errors="coerce")
	spd = pd.to_numeric(df_gps["Spd"], errors="coerce")
	vz = pd.to_numeric(df_gps["VZ"], errors="coerce")
	climb = -vz  # ArduPilot convention: negative VZ means climbing.

	trajectory = pd.DataFrame(
		{"Eeast": e, "North": n, "Up": u, "Spd": spd, "VZ": vz, "ClimbRate": climb}
	).dropna()
	if trajectory.empty or len(trajectory) < 2:
		raise ValueError("df_gps has no valid ENU points to plot")

	return trajectory


def _resolve_color_metric(trajectory: pd.DataFrame, color_by: str) -> tuple[np.ndarray, str, str]:
	color_mode = color_by.strip().lower()
	if color_mode == "ground":
		return trajectory["Spd"].to_numpy(dtype=float), "Ground Speed (m/s)", "Ground Speed"

	if color_mode == "vertical":
		return trajectory["ClimbRate"].to_numpy(dtype=float), "Climb Rate (m/s)", "Climb Rate"

	velocity_color = np.sqrt(
		trajectory["Spd"].to_numpy(dtype=float) ** 2 + trajectory["ClimbRate"].to_numpy(dtype=float) ** 2
	)
	return velocity_color, "Total Speed (m/s)", "Total Speed"


def _build_figure(
	x: np.ndarray,
	y: np.ndarray,
	z: np.ndarray,
	ground_speed: np.ndarray,
	climb_rate: np.ndarray,
	velocity_color: np.ndarray,
	color_title: str,
	plot_title_suffix: str,
):
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

	return fig


def plot_flight_path_3d(
	df_gps: pd.DataFrame,
	output_html: str = "flight_trajectory_enu.html",
	auto_open: bool = False,
	color_by: str = "combined",
):
	"""Build an interactive Plotly 3D trajectory with velocity-based dynamic coloring."""
	_validate_plot_input(df_gps)
	trajectory = _build_trajectory(df_gps)
	velocity_color, color_title, plot_title_suffix = _resolve_color_metric(trajectory, color_by)

	x = np.ravel(trajectory["Eeast"].to_numpy(dtype=float))
	y = np.ravel(trajectory["North"].to_numpy(dtype=float))
	z = np.ravel(trajectory["Up"].to_numpy(dtype=float))
	ground_speed = np.ravel(trajectory["Spd"].to_numpy(dtype=float))
	climb_rate = np.ravel(trajectory["ClimbRate"].to_numpy(dtype=float))

	fig = _build_figure(x, y, z, ground_speed, climb_rate, velocity_color, color_title, plot_title_suffix)

	if output_html:
		fig.write_html(output_html, include_plotlyjs="cdn", auto_open=auto_open)
		print(f"Saved interactive trajectory plot to {output_html}")

	return fig
