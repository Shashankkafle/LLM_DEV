"""
app.py — interactive viewer with a live intersection picker.

Run with:
    bokeh serve --show app.py --args path/to/your_log.json

Falls back to sample_logs.json if no path is given.
"""
import sys

from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import Div, MultiChoice

from data import intersections, load_events, phase_names, to_segments
from viz import timeline_plot

DEFAULT_LOG_PATH = "sample_logs.json"
DEFAULT_NUM_SHOWN = 5  # how many intersections to display when the page first loads


def build_header(log_path, events, segments, all_intersections):
    return Div(text=f"""
        <h2 style="margin-bottom:0;">Traffic Signal Phase Timeline</h2>
        <p style="color:#666;margin-top:4px;">
            Source: <code>{log_path}</code> &mdash;
            {len(events):,} logged events &rarr; {len(segments):,} phase blocks
            across {len(all_intersections)} intersections.
        </p>
    """)


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG_PATH

    events = load_events(log_path)
    segments = to_segments(events)
    all_intersections = intersections(segments)
    all_phases = phase_names(segments)

    default_selection = all_intersections[:DEFAULT_NUM_SHOWN]

    selector = MultiChoice(
        title="Select intersection(s) to display:",
        value=default_selection,
        options=all_intersections,
        width=600,
    )

    plot_area = column(timeline_plot(segments, default_selection, all_phases))

    def on_selection_change(attr, old, new):
        plot_area.children = [timeline_plot(segments, selector.value, all_phases)]

    selector.on_change("value", on_selection_change)

    header = build_header(log_path, events, segments, all_intersections)
    curdoc().add_root(column(header, selector, plot_area))
    curdoc().title = "Traffic Signal Phase Viewer"


main()
