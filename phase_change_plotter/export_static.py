"""
export_static.py — save the timeline as a single shareable HTML file.
No server needed; the file works by itself once opened in a browser.

Usage:
    python export_static.py path/to/log.json -o output.html
    python export_static.py path/to/log.json -o output.html --intersections intersection_1_1 intersection_2_2
"""
import argparse

from bokeh.io import output_file, save
from bokeh.layouts import column
from bokeh.models import Div

from data import intersections, load_events, phase_names, to_segments
from viz import timeline_plot

DEFAULT_NUM_SHOWN = 8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", help="Path to the JSON phase log file")
    parser.add_argument("-o", "--output", default="phase_timeline.html")
    parser.add_argument(
        "--intersections", nargs="*", default=None,
        help=f"Intersection IDs to include (default: first {DEFAULT_NUM_SHOWN} found)",
    )
    args = parser.parse_args()

    events = load_events(args.log_path)
    segments = to_segments(events)
    all_intersections = intersections(segments)
    all_phases = phase_names(segments)

    selection = args.intersections or all_intersections[:DEFAULT_NUM_SHOWN]
    unknown = set(selection) - set(all_intersections)
    if unknown:
        raise ValueError(f"Unknown intersection id(s): {sorted(unknown)}")

    header = Div(text=f"""
        <h2 style="margin-bottom:0;">Traffic Signal Phase Timeline</h2>
        <p style="color:#666;">
            {len(events):,} events &rarr; {len(segments):,} phase blocks &mdash;
            showing {len(selection)} of {len(all_intersections)} intersections.
        </p>
        <p style="font-size:0.85em;color:#888;">
            Drag on the main chart to zoom, scroll to zoom, drag to pan.
            Drag the shaded box below to jump to a time range.
            Click a legend entry to hide/show that phase.
        </p>
    """)

    page = column(header, timeline_plot(segments, selection, all_phases))

    output_file(args.output, title="Traffic Signal Phase Timeline")
    save(page)

    print(f"Saved to {args.output}")
    print(f"Shown: {selection}")
    print(f"All available: {all_intersections}")


if __name__ == "__main__":
    main()
