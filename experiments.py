"""Declarative experiment definitions for the run matrix.

An experiment is a named sweep: the cartesian product of controllers x
simulation configs x seeds x blockage arms. run_matrix.py expands it into
individual runs; build_results.py aggregates their manifests.

Two ideas keep the run data manageable:

  * Aliases. Short names (hz1, c1) resolve to full sumocfg / scenario paths, so
    experiment definitions stay readable and a path change touches one place.

  * Experiment-independent identity. run_identity() distills a run down to what
    actually determines its result (controller, routes, timing, seed, blockage).
    The experiment name is NOT part of it, so the SAME combo run under two
    experiments is recognised as one result -- that is what lets run_matrix
    skip or reuse instead of re-running.
"""
import json
import re
from itertools import product
from pathlib import Path

from configurations import LLM_DEFAULT_PATH
from models_inference.LLM.openrouter_llm import OPENROUTER_PREFIX

# --- aliases -----------------------------------------------------------------

CONFIGS = {
    "hz1": "dataset/llm_light/Hangzhou/4_4/anon_4_4_hangzhou_real_cfphys.sumocfg",
    # Base "real" routes: no vType sigma, so SUMO defaults it to 0.5 -> driving
    # is stochastic and --seed genuinely diverges runs (unlike the cfphys
    # variants, which zero sigma and are seed-invariant). Same 2983-veh demand.
    "hzreal": "dataset/llm_light/Hangzhou/4_4/anon_4_4_hangzhou_real.sumocfg",
    "jinan1": "dataset/llm_light/Jinan/3_4/anon_3_4_jinan_real_cfphys.sumocfg",
}

_HZ_SCENARIOS = "dataset/llm_light/Hangzhou/4_4/scenarios"
BLOCKAGES = {
    "none": None,
    "c1": f"{_HZ_SCENARIOS}/c1_through_west_2_2_600s.json",
    "c2": f"{_HZ_SCENARIOS}/c2_through_west_2_2_1200s.json",
    "c3": f"{_HZ_SCENARIOS}/c3_through_north_2_3_1200s.json",
    "c4": f"{_HZ_SCENARIOS}/c4_two_lane_west_2_2_1200s.json",
}

# controller token -> how run_matrix launches it. The token is what the run's
# manifest records as "controller" (colight/advanced_colight go in as their
# --variant), so identity matching keys off it directly.
CONTROLLERS = {
    "fixedtime": {"script": "runner_baselines.py", "kind": "baseline"},
    "maxpressure": {"script": "runner_baselines.py", "kind": "baseline"},
    "colight": {"script": "runner_colight.py", "kind": "colight"},
    "advanced_colight": {"script": "runner_colight.py", "kind": "colight"},
    "llm": {"script": "runner.py", "kind": "llm"},
}

SHORT_TOKENS = {
    "fixedtime": "ft", "maxpressure": "mp",
    "colight": "colight", "advanced_colight": "advcolight", "llm": "llm",
}


# --- experiments -------------------------------------------------------------

# The one model the whole LLM grid runs on: a custom-fine-tuned, LoRA-merged
# Qwen2.5-14B living outside the repo on the GPU box. runner.py's --llm_path
# takes this directory directly (a merged model dir, not an HF cache folder, so
# there is no snapshots/<hash>/ to descend into); open_llm expands the leading
# ~ at load time. fp16 (~28 GB) fits the A40.
#
# The model IS part of a run's identity (see _identity_fields), so two models
# over the same config/seed/blockage combos are distinct results and share a
# logs tree safely. An "openrouter:<provider>/<model>" value here runs the arm
# against a hosted model instead (see runner.build_llm).
LLM_MODEL_PATH = "~/LLMTSCS-custom_prompts/ft_models/merged/qwen2.5_14b"

EXPERIMENTS = {
    # The blockage lever sweep (C1/C2/C3) on MaxPressure -- cheap, no GPU.
    "mp_blockage_sweep": {
        "controllers": ["maxpressure"],
        "configs": ["hz1"],
        "seeds": [1, 2, 3],
        "blockages": ["none", "c1", "c2", "c3"],
        "steps": 3600,
        "intersection_config": "three_lane",
    },
    # C3 lock-in gate: MP x 3 seeds on clean vs C3. Accept C3 if the C3-vs-none
    # ATT delta is >= ~5x the clean seed SD (build_results' delta_over_clean_spread).
    # CoLight/Adv-CoLight evals join this folder via `--run_group c3_decision`
    # (they eval clean weights, so they run outside the matrix -- see notes).
    "c3_decision": {
        "controllers": ["maxpressure"],
        "configs": ["hz1"],
        "seeds": [1, 2, 3],
        "blockages": ["none", "c3"],
        "steps": 3600,
        "intersection_config": "three_lane",
    },
    # Seed spread on the clean network + FixedTime gridlock canary.
    "baseline_seed_spread": {
        "controllers": ["maxpressure", "fixedtime"],
        "configs": ["hz1"],
        "seeds": [1, 2],
        "blockages": ["none"],
        "steps": 3600,
        "intersection_config": "three_lane",
    },
    # FixedTime seed spread on the base real routes (stochastic driving),
    # clean vs the C3 blockage. Runs on hzreal so the 3 seeds actually diverge.
    "ft_real_normal_c3": {
        "controllers": ["fixedtime"],
        "configs": ["hzreal"],
        "seeds": [1, 2, 3],
        "blockages": ["none", "c3"],
        "steps": 3600,
        "intersection_config": "three_lane",
    },
    # Blockage-campaign baselines: FixedTime + MaxPressure on the real
    # (stochastic) routes, clean vs one blockage arm, 5 seeds.
    # run_blockage_campaign.py runs these, then drops the CoLight evals into the
    # SAME logs/<arm>_campaign/ group so one build_results call tabulates all
    # three controllers together. The two arms are separate experiments (not one
    # sweep over ["none","c2","c3"]) so each campaign is its own self-contained
    # group with its own clean baseline.
    "c2_campaign": {
        "controllers": ["fixedtime", "maxpressure"],
        "configs": ["hzreal"],
        "seeds": [1, 2, 3, 4, 5],
        "blockages": ["none", "c2"],
        "steps": 3600,
        "intersection_config": "three_lane",
    },
    # C3 at the same 5 seeds as C2. Seeds 1-3 of the FixedTime arms already
    # exist in ft_real_normal_c3 and the CoLight evals in colight_c3_eval; both
    # are reused by identity, so this adds seeds 4-5 plus the MaxPressure arm
    # that C3 never had.
    "c3_campaign": {
        "controllers": ["fixedtime", "maxpressure"],
        "configs": ["hzreal"],
        "seeds": [1, 2, 3, 4, 5],
        "blockages": ["none", "c3"],
        "steps": 3600,
        "intersection_config": "three_lane",
    },
    # CoLight + Advanced-CoLight trained on the same real routes, so their
    # clean-network numbers compare directly against ft_real_normal_c3.
    "colight_train_real": {
        "controllers": ["colight", "advanced_colight"],
        "configs": ["hzreal"],
        "seeds": [42],
        "blockages": ["none"],
        "steps": 3600,
        "intersection_config": "three_lane",
        "extra": {"mode": "train_eval", "num_rounds": 100},
    },
    # --- LLM on the real (stochastic) routes: the 12-run grid ---------------
    # Four arms (normal / +text / -text / approach-only), each x 3 seeds, all
    # on the fine-tuned 14B (LLM_MODEL_PATH). Separate experiments (not one
    # sweep) because the per-arm prompt treatment is carried by `extra`
    # (hide_blockage_info, blockage_info_scope), which a single experiment
    # would share across its whole product. hzreal so the seeds diverge; steps
    # and config match ft_real_normal_c3 for a direct baseline comparison.
    #
    # Clean network, no incident. Validates the whole real-routes LLM pipeline
    # end-to-end; run seed 1 of this first.
    "llm_real_normal": {
        "controllers": ["llm"],
        "configs": ["hzreal"],
        "seeds": [1, 2, 3, 4, 5],
        "blockages": ["none"],
        "steps": 3600,
        "intersection_config": "three_lane",
        "extra": {"llm_path": LLM_MODEL_PATH},
    },
    # C3 blockage with the incident text shown in the prompt (the informed arm
    # the prompt audit inspects via decisions.jsonl).
    "llm_real_c3_text": {
        "controllers": ["llm"],
        "configs": ["hzreal"],
        "seeds": [1, 2, 3, 4, 5],
        "blockages": ["c3"],
        "steps": 3600,
        "intersection_config": "three_lane",
        "extra": {"llm_path": LLM_MODEL_PATH},
    },
    # C3 blockage injected physically but kept OUT of the prompt (ablation:
    # does the LLM use the incident text, or only react to the queue numbers?).
    "llm_real_c3_notext": {
        "controllers": ["llm"],
        "configs": ["hzreal"],
        "seeds": [1, 2, 3, 4, 5],
        "blockages": ["c3"],
        "steps": 3600,
        "intersection_config": "three_lane",
        "extra": {"llm_path": LLM_MODEL_PATH, "hide_blockage_info": True},
    },
    # C3 with the incident text WITHHELD from the upstream intersection
    # (intersection_2_4, whose south exit holds the blockage and into which the
    # queue spills) and shown only to the downstream approach intersection
    # (intersection_2_3). scope="approach" == inform the approach intersection
    # only (see runner.py --blockage_info_scope). Isolates whether the upstream
    # intersection's knowledge -- the one that could meter its output into the
    # blocked lane -- is what the informed arm's advantage comes from.
    "llm_real_c3_approach_only": {
        "controllers": ["llm"],
        "configs": ["hzreal"],
        "seeds": [1, 2, 3],
        "blockages": ["c3"],
        "steps": 3600,
        "intersection_config": "three_lane",
        "extra": {"llm_path": LLM_MODEL_PATH, "blockage_info_scope": "approach"},
    },
    # --- LLM on the C2 incident: the +text / -text pair ----------------------
    # Same treatment as the C3 arms above, moved to C2. C2 blocks the West
    # through lane into intersection_2_2 (road_1_2_0_1), so the incident is
    # reported at intersection_2_2 (approach) and intersection_1_2 (downstream
    # exit) -- the reporting pair follows the scenario's lane, nothing here or
    # in the runner names an intersection. Clean baseline for the C2 delta is
    # llm_real_normal (same controller, config, steps and seeds).
    "llm_real_c2_text": {
        "controllers": ["llm"],
        "configs": ["hzreal"],
        "seeds": [1, 2, 3, 4, 5],
        "blockages": ["c2"],
        "steps": 3600,
        "intersection_config": "three_lane",
        "extra": {"llm_path": LLM_MODEL_PATH},
    },
    # C2 injected physically but kept OUT of the prompt (does the LLM use the
    # incident text, or only react to the queue numbers?).
    "llm_real_c2_notext": {
        "controllers": ["llm"],
        "configs": ["hzreal"],
        "seeds": [1, 2, 3, 4, 5],
        "blockages": ["c2"],
        "steps": 3600,
        "intersection_config": "three_lane",
        "extra": {"llm_path": LLM_MODEL_PATH, "hide_blockage_info": True},
    },
    # Clean-network benchmark suite (supersedes run_report_suite.py). CoLight
    "report_suite": {
        "controllers": ["fixedtime", "maxpressure", "colight", "advanced_colight"],
        "configs": ["hz1", "jinan1"],
        "seeds": [42],
        "blockages": ["none"],
        "steps": 3600,
        "intersection_config": "three_lane",
        "extra": {"mode": "train_eval", "num_rounds": 100},
    },
}


def _scenario_name(path):
    if not path:
        return "none"
    try:
        with open(path) as f:
            return json.load(f).get("scenario_name") or Path(path).stem
    except OSError:
        return Path(path).stem


def make_combo(controller, config_alias, seed, blockage_alias, steps,
               intersection_config, extra):
    """Resolve aliases into a fully-specified run description."""
    scenario_path = BLOCKAGES[blockage_alias]
    return {
        "controller": controller,
        "config_alias": config_alias,
        "simulation_config": CONFIGS[config_alias],
        "seed": seed,
        "blockage_alias": blockage_alias,
        "blockage_scenario": scenario_path,
        "blockage_name": _scenario_name(scenario_path),
        "steps": steps,
        "intersection_config": intersection_config,
        "extra": dict(extra or {}),
    }


def expand_experiment(name, overrides=None):
    """Expand an experiment into its list of combos. overrides may replace
    'controllers', 'configs', 'seeds', 'blockages', 'steps', 'num_rounds' or
    'llm_paths', 'max_new_tokens' or 'request_timeout' (for ad-hoc tweaks
    without editing the preset)."""
    if name not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment '{name}'. "
                       f"Known: {sorted(EXPERIMENTS)}")
    exp = dict(EXPERIMENTS[name])
    overrides = overrides or {}
    controllers = overrides.get("controllers") or exp["controllers"]
    configs = overrides.get("configs") or exp.get("configs") or [exp["config"]]
    seeds = overrides.get("seeds") or exp["seeds"]
    blockages = overrides.get("blockages") or exp["blockages"]
    steps = overrides.get("steps") or exp.get("steps", 3600)
    iconf = exp.get("intersection_config", "three_lane")
    extra = dict(exp.get("extra", {}))
    # num_rounds IS part of run identity, so a shortened training is a distinct
    # result -- it never silently pools with the 100-round runs.
    if overrides.get("num_rounds"):
        extra["num_rounds"] = overrides["num_rounds"]

    # Generation budget and request timeout are deliberately NOT part of run
    # identity: a model that needs a bigger budget is a different --llm_path,
    # which identity already carries. Putting them in the key would invalidate
    # every completed run for no gain. Both are recorded in the run manifest.
    for key in ("max_new_tokens", "request_timeout", "reasoning_max_tokens"):
        # "is not None": max_new_tokens=0 means uncapped, not unset.
        if overrides.get(key) is not None:
            extra[key] = overrides[key]

    # --reasoning IS part of run identity (see _identity_fields): thinking on
    # and thinking off are different arms of the same model.
    if overrides.get("reasoning"):
        extra["reasoning"] = overrides["reasoning"]

    # --quantization IS part of run identity for the same reason: 8-bit weights
    # decide differently from full-precision ones.
    if overrides.get("quantization"):
        extra["quantization"] = overrides["quantization"]

    # The model is a sweep dimension for LLM runs, like seeds and blockages: it
    # is part of run identity, so each model is a distinct result that the
    # matrix runs (and can skip/reuse) on its own.
    llm_paths = overrides.get("llm_paths")
    if llm_paths and "llm" not in controllers:
        raise ValueError(
            f"llm_paths was given but experiment '{name}' runs {controllers}; "
            "only the 'llm' controller takes a model.")

    _check_llm_paths(llm_paths)

    _check_aliases(controllers, configs, blockages)
    combos = []
    for controller, config_alias, seed, blk in product(
            controllers, configs, seeds, blockages):
        for llm_path in (llm_paths or [None]) if controller == "llm" else [None]:
            combo_extra = dict(extra)
            if llm_path:
                combo_extra["llm_path"] = llm_path
            combos.append(make_combo(controller, config_alias, seed, blk,
                                     steps, iconf, combo_extra))
    return combos


def _check_aliases(controllers, configs, blockages):
    unknown = ([c for c in controllers if c not in CONTROLLERS]
               + [c for c in configs if c not in CONFIGS]
               + [b for b in blockages if b not in BLOCKAGES])
    if unknown:
        raise KeyError(f"Unknown alias(es): {unknown}")


def _looks_like_local_path(llm_path):
    """A HuggingFace repo id is exactly 'namespace/name'. Anything with a
    backslash, a leading ~ / . / separator, or more than one '/' is meant as a
    filesystem path."""
    return ("\\" in llm_path
            or llm_path.startswith(("~", ".", "/"))
            or llm_path.count("/") > 1)


def _check_llm_paths(llm_paths):
    """Reject an unusable --llm_paths value before any run starts.

    Without this the bad value travels all the way into from_pretrained, which
    treats an unresolvable path as a Hub repo id -- so a mistyped local path
    surfaces as an opaque HFValidationError, once per combo, after each run has
    already booted CUDA.
    """
    for llm_path in llm_paths or []:
        # The hosted server owns which model names it serves.
        if llm_path.startswith(OPENROUTER_PREFIX):
            continue
        if _looks_like_local_path(llm_path):
            resolved = Path(llm_path).expanduser()
            if not resolved.is_dir():
                raise ValueError(
                    f"--llm_path {llm_path!r} is not a directory "
                    f"(looked in {resolved.resolve()}). Local models need an "
                    "absolute or ~-rooted path: the matrix runs runner.py as a "
                    "subprocess, so a path relative to your shell will not "
                    "resolve.")
        elif llm_path.count("/") != 1:
            raise ValueError(
                f"--llm_path {llm_path!r} is neither a local directory nor a "
                "HuggingFace repo id, which must be 'namespace/name'.")


def model_token(llm_path):
    """Short, filesystem-safe tag for a model, for run-dir names and report
    columns: 'openrouter:google/gemma-3-27b-it' -> 'gemma-3-27b-it',
    '~/ft_models/merged/qwen2.5_14b' -> 'qwen2.5_14b'."""
    name = llm_path or ""
    if name.startswith(OPENROUTER_PREFIX):
        name = name[len(OPENROUTER_PREFIX):]
    tail = re.split(r"[\\/]", name.rstrip("\\/"))[-1]
    return re.sub(r"[^A-Za-z0-9._-]", "_", tail)


def combo_slug(combo):
    """Human-friendly run-dir name, e.g. mp_hz1_c2_seed1. Not load-bearing --
    build_results reads identity from the manifest, not this. LLM runs carry a
    model tag so a multi-model sweep is readable on disk."""
    token = SHORT_TOKENS.get(combo["controller"], combo["controller"])
    seed = "dflt" if combo["seed"] is None else combo["seed"]
    slug = f"{token}_{combo['config_alias']}_{combo['blockage_alias']}_seed{seed}"
    llm_path = _combo_llm_path(combo)
    if llm_path:
        slug = f"{slug}_{model_token(llm_path)}"
    reasoning = combo["extra"].get("reasoning")
    if reasoning and reasoning != "auto":
        slug = f"{slug}_think-{reasoning}"
    quantization = combo["extra"].get("quantization")
    if quantization and quantization != "none":
        slug = f"{slug}_{quantization}"
    return slug


# --- identity: what makes two runs the same result ---------------------------

def _identity_fields(controller, simulation_config, intersection_config, steps,
                     seed, blockage_name, hide_info, info_scope, num_rounds,
                     llm_path=None, reasoning=None, quantization=None):
    # llm_path is appended, never inserted: config_key() drops the seed by
    # position (index 4), so the leading fields must keep their offsets.
    return (
        controller,
        Path(simulation_config).name if simulation_config else None,
        intersection_config,
        steps,
        seed,
        blockage_name or "none",
        bool(hide_info),
        info_scope or "both",
        num_rounds,
        llm_path,
        # "auto" normalizes to None so runs from before the flag existed --
        # whose manifests have no reasoning setting -- keep their identity.
        None if reasoning in (None, "auto") else reasoning,
        # Same normalization for the same reason: "none" is the pre-flag
        # behavior, so it must key identically to a manifest that predates it.
        None if quantization in (None, "none") else quantization,
    )


def _combo_llm_path(combo):
    """The model an LLM combo will actually run on. Falls back to runner.py's
    --llm_path default so a combo that leaves it unset still matches the
    manifest of its own completed run (which records the resolved default)."""
    if combo["controller"] != "llm":
        return None
    return combo["extra"].get("llm_path") or LLM_DEFAULT_PATH


def identity_from_combo(combo):
    extra = combo["extra"]
    return _identity_fields(
        combo["controller"], combo["simulation_config"],
        combo["intersection_config"], combo["steps"], combo["seed"],
        combo["blockage_name"], extra.get("hide_blockage_info", False),
        extra.get("blockage_info_scope", "both"), extra.get("num_rounds"),
        _combo_llm_path(combo), extra.get("reasoning"),
        extra.get("quantization"))


def identity_from_manifest(m):
    env = m.get("environment") or {}
    args = m.get("args") or {}
    blk = m.get("blockage") or {}
    return _identity_fields(
        m.get("controller"), env.get("simulation_config"),
        env.get("intersection_config"), env.get("simulation_steps"),
        env.get("seed"), blk.get("scenario_name", "none"),
        blk.get("hide_blockage_info", False),
        blk.get("blockage_info_scope", "both"), args.get("num_rounds"),
        args.get("llm_path"), args.get("reasoning"),
        args.get("quantization"))


def identity_key(identity):
    return "|".join("" if x is None else str(x) for x in identity)


def config_key(identity):
    """Identity with the seed dropped -- groups replicate seeds of one config."""
    fields = list(identity)
    fields[4] = None  # seed is field index 4 in _identity_fields
    return identity_key(fields)
