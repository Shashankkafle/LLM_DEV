# Traffic Signal Phase Timeline Viewer

Visualize how traffic-signal phases change over time, across one or many
intersections, from SUMO/TraCI-style JSON phase logs.

## Why Bokeh

- Draws all phase-bars as one vectorized glyph layer (fast even at tens
  of thousands of bars), not one DOM element per bar.
- Pan/zoom/hover are built-in tools — no custom JS needed.
- All intersection rows share one time axis, so zooming/panning one
  zooms/pans all of them together.

## Files

```
data.py             load JSON -> events -> phase blocks (no plotting code)
viz.py              build the Bokeh chart from phase blocks
app.py              interactive app with a live intersection picker
export_static.py    save a single shareable HTML file, no server needed
make_sample_data.py generates the sample data below, for testing
sample_logs.json    small demo dataset
```

## Usage

**Interactive (recommended)** — pick intersections live in the browser:
```bash
pip install bokeh pandas
bokeh serve --show app.py --args path/to/your_log.json
```

**Static file** — one HTML file you can open or share, no server:
```bash
python export_static.py path/to/your_log.json -o my_timeline.html --intersections intersection_1_1 intersection_2_2
```

**In a notebook:**
```python
from data import load_events, to_segments, intersections, phase_names
from viz import timeline_plot
from bokeh.plotting import show
from bokeh.io import output_notebook

output_notebook()
events = load_events("your_log.json")
segments = to_segments(events)
show(timeline_plot(segments, intersections(segments)[:5], phase_names(segments)))
```

## How it works

1. `load_events` reads the JSON, skips `original_run_details` and any
   key that isn't a real phase record, and returns one row per logged
   event: `timestep, intersection_id, phase_name, phase`.
2. `to_segments` collapses consecutive identical phases (per
   intersection) into single blocks with a `start`/`end` timestep —
   this is what turns 20 repeated `ETWT_GREEN` events into **one** bar
   instead of 20.
3. `timeline_plot` draws one row per intersection and one colored bar
   per phase block, with hover tooltips and a mini overview chart below
   for jumping around long simulations.

Tested with 150,000 events across 36 intersections: loads, compresses to
~43,500 blocks, and renders in well under 3 seconds total.

## Extending

- **More hover info**: add the column in `load_events`, then reference
  it as `@column_name` in `viz.py`'s tooltip list.
- **Different colors**: change `Category20` in `viz.py`.
- **Filter by phase**: add a second `MultiChoice` in `app.py` bound to
  `phase_names(segments)`, and filter `segments` before calling
  `timeline_plot`.
