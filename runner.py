from sumo_env import SumoEnv
from utils.prompt_builder import getPrompt
from models_inference.LLM.open_llm import LLM_Inference
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_name", type=str, default="rough_test")
    parser.add_argument("--simulation_steps", type=int, default=3600)
    parser.add_argument("--simulation_config", type=str, default="simulations/single_intersection/run.sumocfg")
    parser.add_argument("--llm_path", type=str, default="C:/Users/m6722/Research/LLMTSC_SUMO/models/LLMs/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775")
    parser.add_argument("--use_gui", type=bool, default=False)

    return parser.parse_args()

def main(args):
    print("Starting main function...")
    # Initialize SUMO environment
    env = SumoEnv(sumo_config=args.simulation_config, use_gui=args.use_gui)
    # Initialize LLM
    # llm = LLM_Inference(llm_path=args.llm_path)
    # llm.initialize_llm()
    env.start_simulation()
    intersection_states = {}
    intersections = env.get_intersections()
    for step in range(args.simulation_steps):
        for intersection_id in intersections:
            inter_state = env.get_state(intersection_id)
        env.step()

if __name__ == "__main__":
    args = parse_args()
    main(args)    