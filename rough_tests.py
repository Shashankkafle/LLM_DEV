# like a scratchpad script to test out various things without affecting the main codebase.
from models_inference.LLM.open_llm import LLM_Inference
from configurations import LLM_DEFAULT_PATH

if __name__ == "__main__":
    llm_path = LLM_DEFAULT_PATH  # or any other model path you want to test
    llm_inference = LLM_Inference(llm_path)
    llm_inference.initialize_llm()
    
    prompt = "Once upon a time"
    response = llm_inference.inference(prompt)
    print(response)