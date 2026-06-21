"""
viz.py — turn phase segments into a Bokeh chart.

One main function: timeline_plot(segments, intersections, all_phases)

Produces a state-timeline: one row per intersection, one colored bar per
contiguous phase block, hover tooltips, and a mini overview chart below
for quickly navigating long simulations.
"""
import pandas as pd
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, HoverTool, Range1d, RangeTool
from bokeh.palettes import Category20
from bokeh.plotting import figure
from bokeh.transform import factor_cmap

BAR_HEIGHT = 0.8


def _make_source(segments: pd.DataFrame, intersections: list[str]) -> ColumnDataSource:
    """Filter to the selected intersections and add the columns Bokeh's quad glyph needs."""
    df = segments[segments["intersection_id"].isin(intersections)].copy()

    row = {name: i for i, name in enumerate(intersections)}  # top-to-bottom order
    df["y"] = df["intersection_id"].map(row)
    df["top"] = df["y"] + BAR_HEIGHT / 2
    df["bottom"] = df["y"] - BAR_HEIGHT / 2
    df["duration"] = df["end"] - df["start"]

    return ColumnDataSource(df)


def _add_phase_bars(fig, source, all_phases: list[str]):
    """Draw one rectangle per phase block, colored by phase_name."""
    colors = factor_cmap("phase_name", palette=Category20[20], factors=all_phases)
    bars = fig.quad(
        left="start", right="end", top="top", bottom="bottom",
        source=source, fill_color=colors, line_color="white", line_width=0.5,
        legend_field="phase_name",
    )
    fig.add_tools(HoverTool(renderers=[bars], tooltips=[
        ("Intersection", "@intersection_id"),
        ("Phase", "@phase_name"),
        ("Phase string", "@phase"),
        ("Start", "@start"),
        ("End", "@end"),
        ("Duration", "@duration steps"),
    ]))
    return bars


def timeline_plot(
    segments: pd.DataFrame,
    intersections: list[str],
    all_phases: list[str],
    width: int = 1100,
    row_height: int = 50,
):
    """Build the full chart: zoomable timeline on top, mini overview below."""
    if not intersections:
        return column(figure(width=width, height=150, title="Select an intersection to begin"))

    source = _make_source(segments, intersections)
    x_range = Range1d(segments["start"].min(), segments["end"].max())
    y_range = Range1d(-0.5, len(intersections) - 0.5)

    main = figure(
        width=width, height=max(200, row_height * len(intersections)),
        x_range=x_range, y_range=y_range,
        title="Traffic Signal Phase Timeline",
        tools="xpan,box_zoom,wheel_zoom,reset,save",
        active_drag="box_zoom", active_scroll="wheel_zoom",
    )
    _add_phase_bars(main, source, all_phases)

    main.yaxis.ticker = list(range(len(intersections)))
    main.yaxis.major_label_overrides = dict(enumerate(intersections))
    main.ygrid.grid_line_color = None
    main.xaxis.axis_label = "Simulation timestep"
    main.legend.title = "Phase"
    main.legend.click_policy = "hide"  # click a legend entry to hide/show that phase

    # Mini overview chart with a draggable range selector, for quickly
    # jumping to any point in a long simulation.
    mini = figure(
        width=width, height=120, x_range=x_range, y_range=y_range,
        tools="", toolbar_location=None,
    )
    _add_phase_bars(mini, source, all_phases)
    mini.yaxis.visible = False
    mini.ygrid.grid_line_color = None
    mini.legend.visible = False
    mini.xaxis.axis_label = "Drag the shaded box to navigate"

    range_tool = RangeTool(x_range=main.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.15
    mini.add_tools(range_tool)

    return column(main, mini)
