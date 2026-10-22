"""Run FixedTime on hzreal and print TOSI/SOSI only when they are non-zero.

    python tests/test_oversaturation_observer.py [--scenario c3] [--steps 3600]
"""
import argparse
import sys
from pathlib import Path
from unittest import result

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configurations import INTERSECTION_CONFIGS  # noqa: E402
from experiments import BLOCKAGES, CONFIGS  # noqa: E402
from runner_baselines import FixedTimeController  # noqa: E402
from runner_common import setup_run, run_control_loop, build_blockage_manager  # noqa: E402


class NonZeroPrinter:
    def on_interval_closed(self, tracker, result):
        # print(result)
        for target, sosi in result.sosi.items():
            if(result.lane == "road_0_4_0_1"):
                print("trigger")
                print(f"t={result.green_start} lane={result.lane} target={target} "
                      f"tosi={result.tosi} sosi={sosi}")

    def on_green_start(self, tracker):
        pass

    def on_step(self, tracker, step):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=sorted(BLOCKAGES), default="none")
    parser.add_argument("--steps", type=int, default=3600)
    args = parser.parse_args()

    conf = INTERSECTION_CONFIGS["three_lane"]
    scenario_path = BLOCKAGES[args.scenario]
    _, blockage_manager = build_blockage_manager(scenario_path)

    ctx = setup_run(conf, "observer", CONFIGS["hzreal"], {"blockage_scenario": scenario_path},
                    seed=1, blockage_manager=blockage_manager, oversat_listener=NonZeroPrinter(),use_gui=True)
    controller = FixedTimeController(conf)
    run_control_loop(ctx, args.steps, lambda step, tls, handler: controller.choose(tls, handler)[0])


if __name__ == "__main__":
    main()
