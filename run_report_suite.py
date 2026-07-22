"""Run the report benchmark suite: FixedTime, MaxPressure, CoLight, and
Advanced-CoLight on Hangzhou 1 and Jinan 1 (cfphys routes, 30s green,
3600 steps).

Baselines run first (fast, deterministic). CoLight variants then train from
scratch (train_eval: NUM_ROUNDS rounds, then an eval episode scored with the
same MetricsRecorder). Runs are sequential; each writes to its own
logs/<test_name>_<timestamp>/ directory.

Usage:
    python run_report_suite.py
"""
import subprocess
import sys
import time

from configurations import INTERSECTION_CONFIGS

HZ1 = "dataset/llm_light/Hangzhou/4_4/anon_4_4_hangzhou_real_cfphys.sumocfg"
JINAN1 = "dataset/llm_light/Jinan/3_4/anon_3_4_jinan_real_cfphys.sumocfg"
STEPS = "3600"
NUM_ROUNDS = "100"  # CoLight training rounds; lower for a quicker smoke pass
COLIGHT_SEED = "42"

RUNS = [
    # (script, test_name, extra args)
    ("runner_baselines.py", "final_ft30_hz1_cfphys",
     ["--controller", "fixedtime", "--simulation_config", HZ1]),
    ("runner_baselines.py", "final_mp30_hz1_cfphys",
     ["--controller", "maxpressure", "--simulation_config", HZ1]),
    ("runner_baselines.py", "final_ft30_jinan1_cfphys",
     ["--controller", "fixedtime", "--simulation_config", JINAN1]),
    ("runner_baselines.py", "final_mp30_jinan1_cfphys",
     ["--controller", "maxpressure", "--simulation_config", JINAN1]),
    ("runner_colight.py", "final_colight_hz1_cfphys",
     ["--variant", "colight", "--mode", "train_eval",
      "--num_rounds", NUM_ROUNDS, "--seed", COLIGHT_SEED,
      "--simulation_config", HZ1]),
    ("runner_colight.py", "final_advcolight_hz1_cfphys",
     ["--variant", "advanced_colight", "--mode", "train_eval",
      "--num_rounds", NUM_ROUNDS, "--seed", COLIGHT_SEED,
      "--simulation_config", HZ1]),
    ("runner_colight.py", "final_colight_jinan1_cfphys",
     ["--variant", "colight", "--mode", "train_eval",
      "--num_rounds", NUM_ROUNDS, "--seed", COLIGHT_SEED,
      "--simulation_config", JINAN1]),
    ("runner_colight.py", "final_advcolight_jinan1_cfphys",
     ["--variant", "advanced_colight", "--mode", "train_eval",
      "--num_rounds", NUM_ROUNDS, "--seed", COLIGHT_SEED,
      "--simulation_config", JINAN1]),
]


def check_green_duration():
    green = INTERSECTION_CONFIGS["three_lane"]["global_settings"]["default_green_duration"]
    if green != 30:
        sys.exit(
            f"ABORT: three_lane default_green_duration is {green}, expected 30.\n"
            "The report suite is defined for the 30s plan. Fix configurations.py "
            "(60 is only for FixedTime paper-replication runs)."
        )


def main():
    check_green_duration()
    results = []
    for script, test_name, extra in RUNS:
        cmd = [sys.executable, script, "--test_name", test_name,
               "--intersection_config", "three_lane",
               "--simulation_steps", STEPS] + extra
        print(f"\n=== {test_name}\n    {' '.join(cmd)}", flush=True)
        start = time.time()
        code = subprocess.call(cmd)
        minutes = (time.time() - start) / 60
        results.append((test_name, code, minutes))
        print(f"=== {test_name}: exit {code} after {minutes:.1f} min", flush=True)
        if code != 0:
            print("Run failed; continuing with the rest of the suite.", flush=True)

    print("\n===== SUITE SUMMARY =====")
    for test_name, code, minutes in results:
        status = "ok" if code == 0 else f"FAILED (exit {code})"
        print(f"  {test_name:35s} {status:20s} {minutes:7.1f} min")


if __name__ == "__main__":
    main()
