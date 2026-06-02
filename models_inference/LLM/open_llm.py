from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# only for inference, no training, so no need to set up optimizer, scheduler, etc. for LLM
class LLM_Inference:
        def __init__(self,llm_path,max_new_tokens=512):
                self.llm_path = llm_path
                self.max_new_tokens = max_new_tokens

        def initialize_llm(self):
            device_map = "auto"
            # init LLM
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
            )

            self.llm_model.eval()
            # init tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_path,
                padding_side="left",
                padding=True
            )

            self.tokenizer.pad_token_id = 0
            self.test_generation_kwargs = {
                "min_length": -1,
                "top_k": 50,
                "top_p": 1.0,
                "temperature": 0.1,
                "do_sample": True,
                "max_new_tokens": self.max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id
            }

        def inference(self, prompt):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
            with torch.no_grad():
                outputs = self.llm_model.generate(**inputs, **self.test_generation_kwargs)
            response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            return response
