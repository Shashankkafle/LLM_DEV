"""Compare two run dirs file by file, e.g. to show a change is a no-op.

    python tools/diff_runs.py <run_dir_a> <run_dir_b>

CSV files must be byte-identical. final_summary.json and decisions.jsonl are
compared after dropping identity and wall-clock fields, which differ between
any two runs. Every other file is reported as same/differs for information
only. Exits non-zero if a CSV, final_summary.json or decisions.jsonl differs.
"""
import json
import sys
from pathlib import Path

IDENTITY_SUMMARY_KEYS = {"test_name", "controller", "run_started_at", "run_wall_clock_s"}
IDENTITY_DECISION_KEYS = {"timestamp", "controller"}


def strip_keys(record, keys):
    return {key: value for key, value in record.items() if key not in keys}


def first_jsonl_difference(path_a, path_b, ignored_keys):
    lines_a = path_a.read_text().splitlines()
    lines_b = path_b.read_text().splitlines()
    if len(lines_a) != len(lines_b):
        return f"line count {len(lines_a)} vs {len(lines_b)}"
    for number, (line_a, line_b) in enumerate(zip(lines_a, lines_b), start=1):
        if strip_keys(json.loads(line_a), ignored_keys) != strip_keys(json.loads(line_b), ignored_keys):
            return f"first diff at line {number}"
    return None


def files_under(root):
    return {path.relative_to(root) for path in root.rglob("*") if path.is_file()}


def main(dir_a, dir_b):
    root_a, root_b = Path(dir_a), Path(dir_b)
    files_a, files_b = files_under(root_a), files_under(root_b)
    print("only in A:", sorted(map(str, files_a - files_b)))
    print("only in B:", sorted(map(str, files_b - files_a)))
    csv_total = csv_same = 0
    any_difference = False
    for relative in sorted(files_a & files_b):
        path_a, path_b = root_a / relative, root_b / relative
        if relative.suffix == ".csv":
            csv_total += 1
            same = path_a.read_bytes() == path_b.read_bytes()
            csv_same += same
            any_difference |= not same
            print(f"{'SAME' if same else 'DIFF'}  {relative}")
        elif relative.name == "final_summary.json":
            summary_a = json.loads(path_a.read_text())
            summary_b = json.loads(path_b.read_text())
            differing = sorted(key for key in summary_a.keys() | summary_b.keys()
                               if summary_a.get(key) != summary_b.get(key))
            unexpected = [key for key in differing if key not in IDENTITY_SUMMARY_KEYS]
            any_difference |= bool(unexpected)
            print(f"{'SAME' if not unexpected else 'DIFF'}  {relative}  differing keys: {differing}")
        elif relative.name == "decisions.jsonl":
            problem = first_jsonl_difference(path_a, path_b, IDENTITY_DECISION_KEYS)
            any_difference |= problem is not None
            print(f"{'SAME' if problem is None else 'DIFF'}  {relative} "
                  f"(minus timestamp/controller) {problem or ''}")
        else:
            same = path_a.read_bytes() == path_b.read_bytes()
            print(f"{'same' if same else 'differs'}  {relative}  (info)")
    print(f"CSV: {csv_same}/{csv_total} byte-identical")
    return 1 if any_difference else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
