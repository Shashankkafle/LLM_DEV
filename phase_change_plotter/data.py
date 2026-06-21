"""
data.py — load the JSON log and turn it into drawable phase "blocks".

Two steps, two functions:
    load_events(path)    JSON file  -> flat DataFrame, one row per logged event
    to_segments(events)  flat events -> one row per CONTIGUOUS phase block

The second step is what makes consecutive identical phases draw as a
single bar instead of many tiny ones.

Plain pandas only — no plotting code here, so it's easy to test or reuse.
"""
import json

import pandas as pd

# Top-level JSON keys that are metadata, not timesteps.
META_KEYS = {"original_run_details"}


def load_events(path: str) -> pd.DataFrame:
    """Read the JSON log and return one row per logged (timestep, intersection) event."""
    with open(path) as f:
        raw = json.load(f)

    rows = []
    for key, record in raw.items():
        if key in META_KEYS or not isinstance(record, dict):
            continue
        if "intersection_id" not in record or "phase_name" not in record:
            continue
        rows.append({
            "timestep": int(key),
            "intersection_id": record["intersection_id"],
            "phase_name": record["phase_name"],
            "phase": record.get("phase", ""),
        })

    if not rows:
        raise ValueError("No valid phase records found in this JSON file.")

    df = pd.DataFrame(rows).sort_values(["intersection_id", "timestep"])
    return df.reset_index(drop=True)


def to_segments(events: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse consecutive rows with the same phase (per intersection) into
    one block each: {intersection_id, phase_name, phase, start, end}.

    A block's `end` is the timestep where the next block begins. The last
    block per intersection has no "next" event, so it's padded slightly
    just so it remains visible on the chart.
    """
    df = events.copy()

    # A new block starts whenever the intersection or phase changes.
    is_new_block = (
        (df["intersection_id"] != df["intersection_id"].shift())
        | (df["phase_name"] != df["phase_name"].shift())
    )
    df["block"] = is_new_block.cumsum()

    blocks = df.groupby("block").agg(
        intersection_id=("intersection_id", "first"),
        phase_name=("phase_name", "first"),
        phase=("phase", "first"),
        start=("timestep", "first"),
    )

    # end = start of the next block for the same intersection.
    blocks["end"] = blocks.groupby("intersection_id")["start"].shift(-1)
    pad = max(1, int((df["timestep"].max() - df["timestep"].min()) * 0.01))
    blocks["end"] = blocks["end"].fillna(blocks["start"] + pad)

    blocks = blocks.astype({"start": int, "end": int})
    return blocks.sort_values(["intersection_id", "start"]).reset_index(drop=True)


def intersections(segments: pd.DataFrame) -> list[str]:
    return sorted(segments["intersection_id"].unique())


def phase_names(segments: pd.DataFrame) -> list[str]:
    return sorted(segments["phase_name"].unique())
