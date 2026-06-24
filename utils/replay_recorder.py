import json
from pathlib import Path


class ReplayRecorder:
    """
    Streams traffic-light phase changes to a JSONL file -- one event per line,
    appended as the simulation runs. Writing incrementally keeps memory flat and
    leaves a usable (if partial) record on disk if a long run crashes partway.

    Run metadata is written once to a separate sidecar file, so the event stream
    stays a pure list of events with no in-band metadata to filter out on read.

    Files written into record_dir:
      - replay_record.jsonl : one phase-change event per line
      - replay_meta.json    : details needed to replay the run (config, steps...)
    """

    EVENTS_FILENAME = "replay_record.jsonl"
    META_FILENAME = "replay_meta.json"

    def __init__(self, record_dir, meta):
        self.record_dir = Path(record_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.record_dir / self.EVENTS_FILENAME
        self.meta_path = self.record_dir / self.META_FILENAME

        # Truncate any previous stream and write metadata up front, so the run
        # details are on disk even if the simulation later crashes.
        self.events_path.write_text("")
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=4)

    def record_phase_change(self, step, intersection_id, phase, phase_name):
        """
        Appends one intersection's phase change at a given simulation step.

        Several intersections can change phase on the same step; each becomes its
        own line, so nothing is overwritten. The consumer groups lines by step.
        """
        event = {
            "step": step,
            "intersection_id": intersection_id,
            "phase": phase,
            "phase_name": phase_name,
        }
        with open(self.events_path, "a") as f:
            f.write(json.dumps(event) + "\n")
