import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLM_Inference:
    def __init__(self, llm_path):
        self.llm_path = llm_path
        self.model = None
        self.tokenizer = None
        self.model_family = None

    def initialize_llm(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_path, 
            torch_dtype=torch.float16, 
            device_map="auto", 
            trust_remote_code=True
        )
        self.model_family = self._detect_model_family()
        print(f"[Info] Auto-detected model family: {self.model_family}")

    def _detect_model_family(self):
        model_type = getattr(self.model.config, "model_type", "").lower()
        chat_template = getattr(self.tokenizer, "chat_template", "") or ""
        
        if "qwen" in model_type or "<|im_start|>" in chat_template:
            return "chatml"
        if "alpaca" in self.llm_path.lower():
            return "alpaca"
        return "chatml"

    def _format_prompt(self, raw_user_content):
        system_prompt = "You are an expert in traffic management. You can use your knowledge of traffic commonsense to solve this traffic signal control tasks."
        
        if self.model_family == "alpaca":
            return f"{system_prompt}\n\n### Instruction:\n{raw_user_content}\n\n### Response:\n"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_user_content}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def inference(self, raw_prompt):
        formatted_prompt = self._format_prompt(raw_prompt)
        inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.0,
                do_sample=False
            )

        print("outputs:", outputs)
        
        # CORRECTED SLICING HERE
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0, input_length:]
        print("self.tokenizer.decode(generated_tokens, skip_special_tokens=True", self.tokenizer.decode(generated_tokens, skip_special_tokens=True))
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)