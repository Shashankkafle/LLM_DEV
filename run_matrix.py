"""Run an experiment (a controller x config x seed x blockage sweep).

    python run_matrix.py --experiment mp_blockage_sweep
    python run_matrix.py --experiment mp_blockage_sweep --seeds 4 5   # add seeds
    python run_matrix.py --list
    python run_matrix.py --experiment mp_blockage_sweep --dry_run

    # test several models on one arm, one after another
    python run_matrix.py --experiment llm_real_normal --seeds 1 --steps 300 \
        --llm_paths openrouter:google/gemma-3-27b-it \
                    openrouter:google/gemma-3-12b-it

Each combo becomes one run under logs/<experiment>/. Runs are sequential (one
SUMO at a time). For every combo the matrix decides:

  skip   -- a COMPLETED run with this exact identity already exists in this
            experiment. Left as is.
  reuse  -- that identity was completed in ANOTHER experiment; its run dir is
            copied in (with a reused_from.json marker) instead of re-running.
  run    -- nothing matches; launch the runner.

Identity is experiment-independent (see experiments.run_identity), so the same
baseline shared by several experiments is computed once and copied thereafter.
--force re-runs everything regardless.
"""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from configurations import LOGS_DIR_NAME, RUN_MANIFEST_FILENAME
from experiments import (
    CONTROLLERS, EXPERIMENTS, expand_experiment, combo_slug,
    identity_from_combo, identity_from_manifest, identity_key,
)

FINAL_SUMMARY = "final_summary.json"


def build_command(combo, run_group, test_name):
    """The subprocess argv that launches one combo through its runner."""
    spec = CONTROLLERS[combo["controller"]]
    cmd = [sys.executable, spec["script"],
           "--test_name", test_name, "--run_group", run_group,
           "--simulation_config", combo["simulation_config"],
           "--simulation_steps", str(combo["steps"]),
           "--intersection_config", combo["intersection_config"]]
    if combo["seed"] is not None:
        cmd += ["--seed", str(combo["seed"])]
    if combo["blockage_scenario"]:
        cmd += ["--blockage_scenario", combo["blockage_scenario"]]

    extra, kind = combo["extra"], spec["kind"]
    if kind == "baseline":
        cmd += ["--controller", combo["controller"]]
    elif kind == "colight":
        cmd += ["--variant", combo["controller"],
                "--mode", str(extra.get("mode", "train_eval"))]
        if extra.get("num_rounds") is not None:
            cmd += ["--num_rounds", str(extra["num_rounds"])]
        if extra.get("weights_dir"):
            cmd += ["--weights_dir", extra["weights_dir"]]
        if extra.get("eval_round") is not None:
            cmd += ["--eval_round", str(extra["eval_round"])]
        if extra.get("train_blockage_scenario"):
            cmd += ["--train_blockage_scenario", extra["train_blockage_scenario"]]
    elif kind == "llm":
        if extra.get("llm_path"):
            cmd += ["--llm_path", extra["llm_path"]]
        if extra.get("hide_blockage_info"):
            cmd += ["--hide_blockage_info"]
        if extra.get("blockage_info_scope"):
            cmd += ["--blockage_info_scope", extra["blockage_info_scope"]]
        # Batching is a throughput choice, not part of run identity -- these let
        # an experiment force the sequential path or cap the batch, but a
        # batched and a sequential run of the same combo still share an identity.
        if extra.get("sequential"):
            cmd += ["--sequential"]
        if extra.get("max_batch_size"):
            cmd += ["--max_batch_size", str(extra["max_batch_size"])]
        # Generation budget. Not part of run identity: the arm that needs a
        # bigger budget is a different --llm_path, which the identity already
        # carries, so adding these would only invalidate every existing run.
        if extra.get("max_new_tokens"):
            cmd += ["--max_new_tokens", str(extra["max_new_tokens"])]
        if extra.get("request_timeout"):
            cmd += ["--request_timeout", str(extra["request_timeout"])]
    return cmd


def _group_of(run_dir, logs_dir):
    rel = run_dir.relative_to(logs_dir)
    return rel.parts[0] if len(rel.parts) > 1 else ""


def scan_completed_runs(logs_dir):
    """Map identity_key -> list of completed run records, newest first.

    A run counts as completed only when its manifest says so AND a
    final_summary.json exists -- a crashed or half-written run is ignored, so
    the matrix re-runs it rather than skipping or reusing a broken result."""
    by_identity = {}
    for manifest_path in Path(logs_dir).glob(f"**/{RUN_MANIFEST_FILENAME}"):
        run_dir = manifest_path.parent
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        completed = (manifest.get("status") == "completed"
                     and (run_dir / FINAL_SUMMARY).exists())
        if not completed:
            continue
        record = {
            "run_dir": run_dir,
            "group": _group_of(run_dir, Path(logs_dir)),
            "started_at": manifest.get("started_at") or "",
        }
        key = identity_key(identity_from_manifest(manifest))
        by_identity.setdefault(key, []).append(record)
    for records in by_identity.values():
        records.sort(key=lambda r: r["started_at"], reverse=True)
    return by_identity


def decide_action(combo, experiment, by_identity, force):
    """Return (action, source_record) where action is run|skip|reuse."""
    if force:
        return "run", None
    completed = by_identity.get(identity_key(identity_from_combo(combo)), [])
    here = [r for r in completed if r["group"] == experiment]
    if here:
        return "skip", here[0]
    if completed:
        return "reuse", completed[0]
    return "run", None


def _dest_dir(logs_dir, experiment, slug):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = Path(logs_dir) / experiment / f"{slug}_{ts}"
    suffix = 1
    while dest.exists():
        suffix += 1
        dest = Path(logs_dir) / experiment / f"{slug}_{ts}_{suffix}"
    return dest


def reuse_run(source, logs_dir, experiment, slug):
    dest = _dest_dir(logs_dir, experiment, slug)
    shutil.copytree(source["run_dir"], dest)
    (dest / "reused_from.json").write_text(json.dumps({
        "reused_from": str(source["run_dir"]),
        "copied_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    return dest


def launch_run(combo, logs_dir, experiment, slug):
    """Run the combo; return (exit_code, run_dir or None). Locates the produced
    dir as the new logs/<experiment>/<slug>_* that appears across the call."""
    group_dir = Path(logs_dir) / experiment
    pattern = f"{slug}_*"
    before = set(group_dir.glob(pattern)) if group_dir.exists() else set()
    code = subprocess.call(build_command(combo, experiment, slug))
    after = set(group_dir.glob(pattern)) if group_dir.exists() else set()
    new = sorted(after - before)
    return code, (new[-1] if new else None)


def run_is_complete(run_dir):
    if run_dir is None:
        return False
    manifest_path = run_dir / RUN_MANIFEST_FILENAME
    if not manifest_path.exists():
        return False
    try:
        status = json.loads(manifest_path.read_text()).get("status")
    except (OSError, json.JSONDecodeError):
        return False
    return status == "completed" and (run_dir / FINAL_SUMMARY).exists()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", type=str, help="Name from experiments.EXPERIMENTS")
    parser.add_argument("--list", action="store_true", help="List experiments and exit")
    parser.add_argument("--seeds", type=int, nargs="+", help="Override the preset's seeds")
    parser.add_argument("--controllers", type=str, nargs="+", help="Override controllers")
    parser.add_argument("--configs", type=str, nargs="+", help="Override configs")
    parser.add_argument("--blockages", type=str, nargs="+", help="Override blockage arms")
    parser.add_argument("--steps", type=int, help="Override simulation steps (ad-hoc/smoke runs)")
    parser.add_argument("--num_rounds", type=int,
                        help="Override training rounds (CoLight); part of run identity")
    parser.add_argument("--llm_paths", type=str, nargs="+",
                        help="Sweep the LLM arm over these models, one run each, "
                             "in order. Local model dirs and/or "
                             "'openrouter:<provider>/<model>'. Part of run "
                             "identity, so models never pool with each other.")
    parser.add_argument("--max_new_tokens", type=int,
                        help="Override the LLM generation cap (see runner.py). "
                             "Not part of run identity.")
    parser.add_argument("--request_timeout", type=int,
                        help="Override the OpenRouter per-request timeout in "
                             "seconds. Not part of run identity.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run every combo even if a completed run exists")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the plan (run/skip/reuse per combo) without executing")
    parser.add_argument("--logs_dir", type=str, default=LOGS_DIR_NAME)
    return parser.parse_args()


def overrides_from(args):
    return {k: v for k, v in (
        ("seeds", args.seeds), ("controllers", args.controllers),
        ("configs", args.configs), ("blockages", args.blockages),
        ("steps", args.steps), ("num_rounds", args.num_rounds),
        ("llm_paths", args.llm_paths),
        ("max_new_tokens", args.max_new_tokens),
        ("request_timeout", args.request_timeout)) if v}


def main(args):
    if args.list:
        for name, exp in EXPERIMENTS.items():
            print(f"  {name}: {exp}")
        return
    if not args.experiment:
        sys.exit("Pass --experiment <name> (or --list). Known: "
                 f"{sorted(EXPERIMENTS)}")

    combos = expand_experiment(args.experiment, overrides_from(args))
    by_identity = scan_completed_runs(args.logs_dir)
    print(f"Experiment '{args.experiment}': {len(combos)} combos "
          f"({'DRY RUN' if args.dry_run else 'running'})\n")

    tally = {"run": 0, "skip": 0, "reuse": 0, "failed": 0}
    for combo in combos:
        slug = combo_slug(combo)
        action, source = decide_action(combo, args.experiment, by_identity, args.force)

        if action == "skip":
            print(f"  skip   {slug:28s} (done: {source['run_dir']})")
            tally["skip"] += 1
        elif action == "reuse":
            if args.dry_run:
                print(f"  reuse  {slug:28s} <- {source['run_dir']}")
            else:
                dest = reuse_run(source, args.logs_dir, args.experiment, slug)
                print(f"  reuse  {slug:28s} <- {source['run_dir']} -> {dest}")
            tally["reuse"] += 1
        else:  # run
            if args.dry_run:
                print(f"  run    {slug:28s} {' '.join(build_command(combo, args.experiment, slug))}")
                tally["run"] += 1
                continue
            print(f"  run    {slug:28s} ...", flush=True)
            code, run_dir = launch_run(combo, args.logs_dir, args.experiment, slug)
            if code == 0 and run_is_complete(run_dir):
                print(f"         -> ok: {run_dir}")
                tally["run"] += 1
                # so a later combo in THIS sweep can reuse it
                by_identity.setdefault(
                    identity_key(identity_from_combo(combo)), []).insert(0, {
                        "run_dir": run_dir, "group": args.experiment,
                        "started_at": datetime.now().isoformat()})
            else:
                print(f"         -> FAILED (exit {code}, dir {run_dir})")
                tally["failed"] += 1

    print(f"\nSummary: {tally['run']} run, {tally['skip']} skipped, "
          f"{tally['reuse']} reused, {tally['failed']} failed.")
    if not args.dry_run:
        print("Aggregate with:  python build_results.py --experiment " + args.experiment)


if __name__ == "__main__":
    main(parse_args())
