"""One trigger, the whole blockage campaign: FixedTime, MaxPressure and CoLight
on the real hzreal routes, clean vs each blockage arm, seeds 1-5.

By default it runs both arms back to back -- C2 then C3 -- so the two land at
the same 5 seeds with the same three controllers.

Nothing is retrained. The script locates the CoLight weights already trained on
this network and flow (logs/colight_train_real/, the ones the original C3 evals
used) and runs the same zero-shot eval against each arm, so every number in
both campaigns comes from one policy. Training happens only if no such weights
exist, and --skip_train forbids even that.

Reuse is on throughout: any run whose identity was already computed in an
earlier experiment is copied in rather than recomputed. In practice that means
seeds 1-3 of the FixedTime C3 arms come from ft_real_normal_c3, seeds 1-3 of
the CoLight C3 evals from colight_c3_eval, and the new work is seeds 4-5 plus
the MaxPressure arm C3 never had.

Each arm lands in its own logs/<arm>_campaign/ group, so one build_results call
per arm tabulates all three controllers side by side (the clean arm is what
gives each blockage arm its ATT delta, so it is part of the campaign, not
assumed).

Everything is sequential -- one SUMO at a time, start it and walk away.

Weights are resolved once up front, before any arm runs: look in
logs/<train_group>/ first, then anywhere under logs/ for a completed training
of this variant on the same sumocfg. num_rounds and the scored round come from
the training metadata, so these evals carry the same identity fields as the
original C3 ones and land in the same build_results row family. Sharing one
resolution across arms is what keeps C2 and C3 comparable.

Then, per arm:

  1. Baselines. run_matrix's `<arm>_campaign` experiment: {fixedtime,
     maxpressure} x seeds 1-5 x {none, arm} = 20 combos. One identity is
     guarded against reuse first; see IDENTITY_COLLISIONS below.
  2. CoLight eval. seeds 1-5 x {none, arm}, one greedy eval each, with the same
     skip/reuse-by-identity treatment run_matrix gives baselines.

Re-running the script resumes it: anything already finished is skipped.

    python run_blockage_campaign.py                   # both arms, all phases
    python run_blockage_campaign.py --force           # compute everything fresh
    python run_blockage_campaign.py --arms c3         # C3 only
    python run_blockage_campaign.py --dry_run         # print the plan
    python run_blockage_campaign.py --seeds 4 5       # override the seed set
    python run_blockage_campaign.py --skip_baselines  # CoLight only
    python run_blockage_campaign.py --skip_colight    # baselines only
    python run_blockage_campaign.py --weights_dir logs/colight_train_real/colight_.../weights
    python run_blockage_campaign.py --variants colight advanced_colight

Aggregate with:
    python build_results.py --experiment c2_campaign
    python build_results.py --experiment c3_campaign
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from configurations import (
    LOGS_DIR_NAME,
    COLIGHT_WEIGHTS_DIR_NAME,
    COLIGHT_MODEL_METADATA_FILENAME,
)
from experiments import (
    BLOCKAGES, CONFIGS, EXPERIMENTS, SHORT_TOKENS, combo_slug, make_combo,
)
from run_matrix import decide_action, reuse_run, scan_completed_runs

TRAIN_EXPERIMENT = "colight_train_real"

DEFAULT_VARIANTS = ["colight"]
SEEDS = [1, 2, 3, 4, 5]


def group_for(arm):
    """One log group and one run_matrix experiment per arm."""
    return f"{arm}_campaign"


# An arm is campaignable once experiments.py defines its <arm>_campaign sweep.
CAMPAIGN_ARMS = [arm for arm in BLOCKAGES
                 if arm != "none" and group_for(arm) in EXPERIMENTS]


def eval_blockages(arm):
    return ["none", arm]

# Combos an older run matches by identity without being a valid substitute.
# Green duration is not part of run identity, so the 60s-green regression run
# regress_ft_hz1_60g_orig_s1 looks like a clean 30s-green ft/hzreal/none/seed1
# to the matrix. When one of these IS what the sweep would reuse, it is
# recomputed (--force) first and the sweep then skips it; when a good run wins
# the reuse instead, nothing is forced -- so reuse stays on everywhere else.
IDENTITY_COLLISIONS = [
    {"controller": "fixedtime", "blockage": "none", "seed": 1,
     "bad_run": "regress_ft_hz1_60g_orig_s1"},
]

CONFIG_ALIAS = "hzreal"
SIMULATION_CONFIG = CONFIGS[CONFIG_ALIAS]
STEPS = 3600
INTERSECTION_CONFIG = "three_lane"


def call(cmd, dry_run):
    print(f"  $ {' '.join(cmd)}", flush=True)
    return 0 if dry_run else subprocess.call(cmd)


# --- phase 1: baselines ------------------------------------------------------

def matrix_cmd(experiment, dry_run, *extra):
    cmd = [sys.executable, "-u", "run_matrix.py", "--experiment", experiment]
    cmd += [str(a) for a in extra]
    if dry_run:
        cmd.append("--dry_run")
    return cmd


def would_reuse_bad_run(collision, arm, by_identity):
    """True if the sweep would satisfy this combo with the known-bad run."""
    combo = make_combo(collision["controller"], CONFIG_ALIAS,
                       collision["seed"], collision["blockage"], STEPS,
                       INTERSECTION_CONFIG, {})
    action, source = decide_action(combo, group_for(arm), by_identity,
                                   force=False)
    return action == "reuse" and collision["bad_run"] in str(source["run_dir"])


def run_collision_guards(arm, seeds, dry_run):
    """Recompute any combo the sweep would satisfy with a known-bad older run."""
    by_identity = scan_completed_runs(LOGS_DIR_NAME)
    for collision in IDENTITY_COLLISIONS:
        label = (f"{collision['controller']}/{collision['blockage']}"
                 f"/seed{collision['seed']}")
        if collision["seed"] not in seeds:
            continue
        if not would_reuse_bad_run(collision, arm, by_identity):
            print(f"  guard  {label}: nothing to guard against.")
            continue
        print(f"  guard  {label}: forcing a fresh run "
              f"({collision['bad_run']} shares its identity).")
        cmd = matrix_cmd(group_for(arm), dry_run,
                         "--controllers", collision["controller"],
                         "--blockages", collision["blockage"],
                         "--seeds", collision["seed"], "--force")
        code = call(cmd, dry_run=False)
        if code != 0:
            return code
        if dry_run:
            print("  guard  (a dry run writes nothing, so the sweep below still "
                  "shows this combo as 'reuse'; in a real run it is 'skip'.)")
    return 0


def run_baselines(arm, seeds, dry_run, force):
    """FixedTime + MaxPressure x seeds x {none, arm}, via the run matrix."""
    if force:
        # Nothing is reused, so there is no bad reuse to guard against.
        return call(matrix_cmd(group_for(arm), dry_run, "--seeds", *seeds,
                               "--force"), dry_run=False)
    code = run_collision_guards(arm, seeds, dry_run)
    if code != 0:
        return code
    return call(matrix_cmd(group_for(arm), dry_run, "--seeds", *seeds),
                dry_run=False)


# --- phase 2: locate the already-trained weights -----------------------------

def read_weights_dir(weights_dir, variant=None):
    """(weights_dir, num_rounds, eval_round) for a finished training, else None.

    num_rounds and the scored round come from training_metadata.json: num_rounds
    is an identity field, so reading it keeps these evals in the same
    build_results family as the C3 ones, and the scored round is the last one
    actually written to disk (not one the training never reached).
    """
    weights_dir = Path(weights_dir)
    metadata_path = weights_dir / COLIGHT_MODEL_METADATA_FILENAME
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if variant and metadata.get("controller") != variant:
        return None
    rounds_saved = metadata.get("weights", {}).get("rounds_saved") or []
    if not rounds_saved:
        return None
    num_rounds = metadata.get("training", {}).get("num_rounds", len(rounds_saved))
    return weights_dir, num_rounds, max(rounds_saved)


def _trained_on_this_network(weights_dir):
    metadata = json.loads((weights_dir / COLIGHT_MODEL_METADATA_FILENAME).read_text())
    trained_config = (metadata.get("network") or {}).get("simulation_config") or ""
    return Path(trained_config).name == Path(SIMULATION_CONFIG).name


def find_in_group(variant, train_group):
    group_dir = Path(LOGS_DIR_NAME) / train_group
    for run_dir in sorted(group_dir.glob(f"{SHORT_TOKENS[variant]}_*"), reverse=True):
        found = read_weights_dir(run_dir / COLIGHT_WEIGHTS_DIR_NAME, variant)
        if found:
            return found
    return None


def find_anywhere(variant):
    """Fallback: any completed training of this variant on this sumocfg.

    Covers weights that live outside the expected group -- a training carried
    over from the C3 campaign under a different run-group name, say.
    """
    candidates = []
    for metadata_path in Path(LOGS_DIR_NAME).glob(
            f"**/{COLIGHT_WEIGHTS_DIR_NAME}/{COLIGHT_MODEL_METADATA_FILENAME}"):
        found = read_weights_dir(metadata_path.parent, variant)
        if found and _trained_on_this_network(metadata_path.parent):
            candidates.append(found)
    if not candidates:
        return None
    return max(candidates, key=lambda f: str(f[0]))


def find_trained_weights(variant, train_group):
    return find_in_group(variant, train_group) or find_anywhere(variant)


def run_training(variants, num_rounds, dry_run, force):
    """Train only the variants that are actually missing.

    colight_train_real defines both colight and advanced_colight, and each
    training is a long job -- passing --controllers keeps a single-variant
    campaign from paying for the other one. Without --force the matrix skips a
    training that already completed, so an interrupted campaign resumes into
    the eval phase instead of retraining.
    """
    extra = ["--controllers", *variants]
    if num_rounds:
        extra += ["--num_rounds", str(num_rounds)]
    if force:
        extra.append("--force")
    return call(matrix_cmd(TRAIN_EXPERIMENT, dry_run, *extra), dry_run=False)


DEFAULT_TRAIN_ROUNDS = EXPERIMENTS[TRAIN_EXPERIMENT]["extra"]["num_rounds"]


def _with_placeholders(variants, weights_by_variant, num_rounds):
    """Stand-in weights so --dry_run can still show the eval plan."""
    rounds = num_rounds or DEFAULT_TRAIN_ROUNDS
    placeholder = (Path("logs/<train_group>/<variant>_.../weights"),
                   rounds, rounds - 1)
    return {v: weights_by_variant.get(v) or placeholder for v in variants}


def weights_from_flag(variants, weights_dir):
    if len(variants) != 1:
        sys.exit("--weights_dir names one training, so pass exactly one "
                 f"--variants entry (got {variants}).")
    variant = variants[0]
    found = read_weights_dir(weights_dir, variant)
    if not found:
        sys.exit(f"No usable {variant} weights at {weights_dir} "
                 f"(need {COLIGHT_MODEL_METADATA_FILENAME} with rounds_saved).")
    return {variant: found}


def resolve_weights(variants, args):
    """Weights for every variant: reuse what is on disk, train only if forced to."""
    if args.weights_dir:
        return weights_from_flag(variants, args.weights_dir)

    weights_by_variant = {} if args.force else {
        v: find_trained_weights(v, args.train_group) for v in variants}
    missing = [v for v in variants if weights_by_variant.get(v) is None]
    if not missing:
        print(f"[weights] reusing existing training for {variants}; "
              f"nothing to train.")
        return weights_by_variant
    if args.force:
        print("[weights] --force: training from scratch, ignoring any "
              "existing weights.")

    if args.skip_train:
        if not args.dry_run:
            sys.exit(f"No trained weights found for {missing} (searched "
                     f"logs/{args.train_group}/ then all of logs/) and "
                     f"--skip_train was passed.")
        print(f"[dry_run] no weights for {missing}; "
              f"showing the eval plan with placeholder paths.")
        return _with_placeholders(variants, weights_by_variant,
                                  args.train_rounds)

    rounds = args.train_rounds or DEFAULT_TRAIN_ROUNDS
    print(f"[weights] training {len(missing)} x {rounds} rounds: {missing}")
    code = run_training(missing, args.train_rounds, args.dry_run, args.force)
    if code != 0:
        sys.exit(f"Training failed (exit {code}); not evaluating.")
    if args.dry_run:
        return _with_placeholders(variants, weights_by_variant,
                                  args.train_rounds)

    weights_by_variant = {v: find_trained_weights(v, args.train_group)
                          for v in variants}
    still_missing = [v for v, w in weights_by_variant.items() if w is None]
    if still_missing:
        sys.exit(f"Training finished but no weights found for {still_missing} "
                 f"under logs/{args.train_group}/.")
    return weights_by_variant


# --- phase 3: eval -----------------------------------------------------------

def eval_combo(variant, seed, blockage_alias, num_rounds):
    """The run-matrix combo this eval corresponds to -- used for identity, so a
    result already computed elsewhere is copied in instead of recomputed."""
    return make_combo(variant, CONFIG_ALIAS, seed, blockage_alias, STEPS,
                      INTERSECTION_CONFIG, {"num_rounds": num_rounds})


def build_eval_cmd(variant, weights, combo, slug, arm):
    weights_dir, num_rounds, eval_round = weights
    cmd = [sys.executable, "runner_colight.py",
           "--test_name", slug, "--run_group", group_for(arm),
           "--variant", variant, "--mode", "eval",
           "--num_rounds", str(num_rounds), "--eval_round", str(eval_round),
           "--weights_dir", str(weights_dir),
           "--simulation_config", SIMULATION_CONFIG,
           "--simulation_steps", str(STEPS),
           "--intersection_config", INTERSECTION_CONFIG,
           "--seed", str(combo["seed"])]
    if combo["blockage_scenario"]:
        cmd += ["--blockage_scenario", combo["blockage_scenario"]]
    return cmd


def run_one_eval(variant, weights, combo, arm, by_identity, dry_run, force):
    """Return the action taken: skip | reuse | run | failed."""
    slug = combo_slug(combo)
    group = group_for(arm)
    action, source = decide_action(combo, group, by_identity, force)

    if action == "skip":
        print(f"  skip   {slug:28s} (done: {source['run_dir']})")
        return "skip"
    if action == "reuse":
        if dry_run:
            print(f"  reuse  {slug:28s} <- {source['run_dir']}")
        else:
            dest = reuse_run(source, LOGS_DIR_NAME, group, slug)
            print(f"  reuse  {slug:28s} <- {source['run_dir']} -> {dest}")
        return "reuse"

    print(f"  run    {slug:28s}")
    code = call(build_eval_cmd(variant, weights, combo, slug, arm), dry_run)
    if dry_run:
        return "run"
    print(f"         -> exit {code}")
    return "run" if code == 0 else "failed"


def run_evals(weights_by_variant, arm, seeds, dry_run, force):
    by_identity = {} if force else scan_completed_runs(LOGS_DIR_NAME)
    tally = {"run": 0, "skip": 0, "reuse": 0, "failed": 0}
    for variant, weights in weights_by_variant.items():
        weights_dir, num_rounds, eval_round = weights
        print(f"\n[eval] {variant}: weights={weights_dir} "
              f"num_rounds={num_rounds} eval_round={eval_round}")
        for seed in seeds:
            for blockage_alias in eval_blockages(arm):
                combo = eval_combo(variant, seed, blockage_alias, num_rounds)
                action = run_one_eval(variant, weights, combo, arm,
                                      by_identity, dry_run, force)
                tally[action] += 1
    return tally


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", default=CAMPAIGN_ARMS,
                        choices=CAMPAIGN_ARMS,
                        help=f"Blockage arms to campaign, in order "
                             f"(default: {CAMPAIGN_ARMS}).")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                        choices=["colight", "advanced_colight"],
                        help="CoLight variants to evaluate (default: colight).")
    parser.add_argument("--weights_dir",
                        help="Use this trained weights dir directly "
                             "(one --variants entry only).")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help=f"Seeds for every phase (default: {SEEDS}).")
    parser.add_argument("--train_group", default=TRAIN_EXPERIMENT,
                        help="Log group to look for trained weights in first.")
    parser.add_argument("--train_rounds", type=int,
                        help=f"Training rounds if a training is needed "
                             f"(default: {DEFAULT_TRAIN_ROUNDS}). Fewer rounds "
                             f"is a different result, not a cheaper one: "
                             f"num_rounds is part of run identity, so these "
                             f"runs never pool with 100-round ones.")
    parser.add_argument("--force", action="store_true",
                        help="Compute everything from scratch: no reuse of "
                             "earlier results, no skipping of finished runs, "
                             "and the CoLight training redone. Drop this flag "
                             "when resuming an interrupted campaign.")
    parser.add_argument("--skip_baselines", action="store_true",
                        help="Skip phase 1 (FixedTime + MaxPressure).")
    parser.add_argument("--skip_colight", action="store_true",
                        help="Skip phases 2 and 3 (CoLight).")
    parser.add_argument("--skip_train", action="store_true",
                        help="Never train; fail if no weights are found.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print every phase without running anything.")
    return parser.parse_args()


def run_arm(arm, weights_by_variant, args):
    seeds = args.seeds
    print(f"\n########## {arm.upper()} campaign "
          f"(logs/{group_for(arm)}/) ##########")

    if not args.skip_baselines:
        print(f"\n=== baselines: FixedTime + MaxPressure "
              f"(seeds {seeds} x none/{arm}) ===")
        code = run_baselines(arm, seeds, args.dry_run, args.force)
        if code != 0:
            sys.exit(f"[{arm}] baseline phase failed (exit {code}).")

    if args.skip_colight:
        return
    print(f"\n=== CoLight eval (seeds {seeds} x none/{arm}) ===")
    tally = run_evals(weights_by_variant, arm, seeds, args.dry_run, args.force)
    if not args.dry_run:
        print(f"\n[{arm}] CoLight evals: {tally['run']} run, "
              f"{tally['reuse']} reused, {tally['skip']} skipped, "
              f"{tally['failed']} failed.")


def main(args):
    # Weights are resolved once and shared by every arm: the whole point is
    # that one policy meets each blockage, so the arms stay comparable.
    weights_by_variant = None
    if not args.skip_colight:
        print("=== CoLight weights (resolved once, shared by every arm) ===")
        weights_by_variant = resolve_weights(args.variants, args)

    for arm in args.arms:
        run_arm(arm, weights_by_variant, args)

    if not args.dry_run:
        print("\nAggregate with:")
        for arm in args.arms:
            print(f"  python build_results.py --experiment {group_for(arm)}")


if __name__ == "__main__":
    main(parse_args())
