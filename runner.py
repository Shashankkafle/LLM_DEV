from sumo_env import SumoEnv
# from utils.prompt_builder import getPrompt
from models_inference.LLM.open_llm import LLM_Inference
from utils.phase_handler import PhaseHandler
from configurations import INTERSECTION_CONFIG as conf
# from configurations import green_phases
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
    llm = LLM_Inference(llm_path=args.llm_path)
    llm.initialize_llm()
    # Initialize phase handler
    env.start_simulation()
    intersection_states = {}
    intersection_phase_handlers = {}
    intersections = env.get_intersections()
    for intersection_id in intersections:
        intersection_phase_handlers[intersection_id] = PhaseHandler( env=env, conf=conf, intersection_id=intersection_id, start_phase='ETWT')
    for step in range(args.simulation_steps):
        print(f"Simulation step: {step}")
        env.step()
        for intersection_id, handler in intersection_phase_handlers.items():
            handler.step()
            print(f"Intersection {intersection_id} - Current Phase: {handler.current_phase}, Phase Type: {handler.current_phase_type}, Duration Remaining: {handler.duration}")
            if handler.switch_phase:
                
                state_data = env.get_state(intersection_id) 
                
                # # Build the prompt for the LLM
                # prompt = getPrompt(
                #     state=state_data, 
                #     current_phase=handler.current_phase, 
                #     available_phases=conf["phases"].keys()
                # )
                
                # Query the LLM for the next action 
                # llm_output = llm.generate(prompt) 
                
                # Parse the LLM output to get the next phase (and potentially duration)
                # next_phase = llm_output.strip()
                next_phase =  (list(conf["phases"].keys()))[step % len(conf["phases"].keys())]  # Placeholder for LLM output, cycling through phases for testing
                duration = 30 # Defaulting to 30, or extract this from the LLM response
                
                # Fallback mechanism if the LLM hallucinates an invalid phase
                if next_phase not in conf["phases"]:
                    print(f"[Warning] Invalid phase '{next_phase}' generated at step {step}. Defaulting to current phase.")
                    next_phase = handler.current_phase


                print(f"Step {step}: Intersection {intersection_id} - Current Phase: {handler.current_phase}, Next Phase: {next_phase}, Duration: {duration}")
                # Apply the decision to the SUMO environment
                
                print(f"Activating phase {next_phase} for intersection {intersection_id} with duration {duration}")
                # Reset and update the phase handler
                handler.activate_phase(next_phase)



        


if __name__ == "__main__":
    args = parse_args()
    main(args)    