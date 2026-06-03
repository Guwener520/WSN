"""Interactive Plotly dashboard for WSN deployment analysis."""

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp


def interactive_dashboard(
    env,
    node_positions: np.ndarray | None = None,
    title: str = "WSN Coverage Dashboard",
) -> go.Figure:
    """Create an interactive Plotly figure with node positions and coverage heatmap.

    Args:
        env: WSNEnvironment instance.
        node_positions: (N, 2) array. Uses env.node_positions if None.
        title: dashboard title.

    Returns:
        plotly.graph_objects.Figure with two subplots:
        - Left: node deployment with sensing circles.
        - Right: coverage count heatmap.
    """
    if node_positions is None:
        node_positions = env.node_positions
    if node_positions is None:
        raise ValueError("No node positions available.")
    env.set_positions(node_positions)

    fig = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=("Node Deployment", "Coverage Heatmap"),
        horizontal_spacing=0.12,
    )

    # --- Left: node deployment ---
    # Draw sensing circles
    theta = np.linspace(0, 2 * np.pi, 60)
    for idx, (x, y) in enumerate(node_positions):
        cx = x + env.sensing_radius * np.cos(theta)
        cy = y + env.sensing_radius * np.sin(theta)
        fig.add_trace(
            go.Scatter(
                x=cx, y=cy,
                mode="lines",
                line=dict(color="blue", width=0.5),
                opacity=0.15,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=node_positions[:, 0],
            y=node_positions[:, 1],
            mode="markers",
            marker=dict(color="red", size=8),
            name="Nodes",
            hovertemplate="Node %{pointIndex}<br>x=%{x:.1f}, y=%{y:.1f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # --- Right: coverage heatmap ---
    cov = env.coverage_matrix()
    coverage_counts = np.sum(cov, axis=0).reshape(env.grid_y.shape)

    fig.add_trace(
        go.Heatmap(
            x=env.grid_x[0, :],
            y=env.grid_y[:, 0],
            z=coverage_counts,
            colorscale="YlOrRd",
            zmin=0,
            colorbar=dict(title="Coverage count"),
            name="Coverage",
            hovertemplate="x=%{x:.0f}, y=%{y:.0f}<br>Coverage: %{z}<extra></extra>",
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="X", range=[0, env.width], row=1, col=1)
    fig.update_yaxes(title_text="Y", range=[0, env.height], row=1, col=1)
    fig.update_xaxes(title_text="X", range=[0, env.width], row=1, col=2)
    fig.update_yaxes(title_text="Y", range=[0, env.height], row=1, col=2)

    fig.update_layout(
        title=title,
        width=1200,
        height=550,
        showlegend=False,
    )

    return fig
