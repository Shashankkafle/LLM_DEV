# like a scratchpad script to test out various things without affecting the main codebase.
from models_inference.LLM.open_llm import LLM_Inference

if __name__ == "__main__":
    llm_path = r"C:/Users/m6722/Research/LLMTSC_SUMO/models/LLMs/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"  # or any other model path you want to test
    llm_inference = LLM_Inference(llm_path)
    llm_inference.initialize_llm()
    
    prompt = "Once upon a time"
    response = llm_inference.inference(prompt)
    print(response)