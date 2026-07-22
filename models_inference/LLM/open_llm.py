import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from configurations import (
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    LLM_DO_SAMPLE,
    LLM_SYSTEM_PROMPT,
)


class LLM_Inference:
    def __init__(self, llm_path):
        self.llm_path_arg = llm_path
        self.llm_path = self._resolve_snapshot_path(llm_path)
        self.model = None
        self.tokenizer = None
        self.model_family = None
        self._logged_first_prompt = False
        # Set by every inference() call, read by the runner's recorder.
        self.last_usage = None
        self.last_formatted_prompt = None

    @staticmethod
    def _resolve_snapshot_path(llm_path):
        # Expand a leading ~ ourselves: the runner is launched with a list argv
        # (no shell), so a "~/..." path arrives literal and from_pretrained
        # would treat ~ as a real directory name.
        llm_path = os.path.expanduser(llm_path)
        # A HF cache folder (models--Org--Name) holds the real files under
        # snapshots/<hash>/ — point transformers at that inner directory.
        snapshots_dir = os.path.join(llm_path, "snapshots")
        if os.path.isdir(snapshots_dir):
            hashes = sorted(os.listdir(snapshots_dir))
            if hashes:
                return os.path.join(snapshots_dir, hashes[0])
        return llm_path

    def initialize_llm(self):
        if torch.cuda.is_available():
            print(f"[Info] CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("[Warning] CUDA not available, running on CPU")

        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        device_map = getattr(self.model, "hf_device_map", None)
        if device_map is not None:
            print(f"[Info] Model device map: {device_map}")
        else:
            print(f"[Info] No hf_device_map; model on: {self.model.device}")
        self.model_family = self._detect_model_family()
        print(f"[Info] Auto-detected model family: {self.model_family}")

    def describe(self):
        """Model + generation config for the run manifest, so a run record
        pins down exactly which model, wrapping, and decoding settings
        produced its decisions. Call after initialize_llm()."""
        device_map = getattr(self.model, "hf_device_map", None)
        return {
            "llm_path_arg": self.llm_path_arg,
            "resolved_snapshot_path": self.llm_path,
            "model_family": self.model_family,
            "torch_dtype": str(getattr(self.model, "dtype", None)),
            "device_map": ({k: str(v) for k, v in device_map.items()}
                           if device_map else str(self.model.device)),
            "generation": {
                "max_new_tokens": LLM_MAX_NEW_TOKENS,
                "temperature": LLM_TEMPERATURE,
                "do_sample": LLM_DO_SAMPLE,
            },
            "system_prompt": LLM_SYSTEM_PROMPT,
            "chat_template_present": bool(getattr(self.tokenizer, "chat_template", None)),
        }

    def _detect_model_family(self):
        model_type = getattr(self.model.config, "model_type", "").lower()
        chat_template = getattr(self.tokenizer, "chat_template", "") or ""
        
        if "qwen" in model_type or "<|im_start|>" in chat_template:
            return "chatml"
        if "alpaca" in self.llm_path.lower():
            return "alpaca"
        return "chatml"

    def _format_prompt(self, raw_user_content):
        system_prompt = LLM_SYSTEM_PROMPT

        if self.model_family == "alpaca":
            alpaca = f"{system_prompt}\n\n### Instruction:\n{raw_user_content}\n\n### Response:\n"
            return self._log_first_prompt(alpaca)

        # Only reach here for chatml — guard against missing template
        if not getattr(self.tokenizer, "chat_template", None):
            print("[Warning] chatml family detected but no chat_template found. Falling back to Alpaca.")
            alpaca = f"{system_prompt}\n\n### Instruction:\n{raw_user_content}\n\n### Response:\n"
            return self._log_first_prompt(alpaca)

        # Prefer a proper system + user turn. Some fine-tuned chat templates
        # (seen on Llama2-derived models) reject a separate "system" role; if so,
        # fold the system text into the user turn rather than crashing mid-run.
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": raw_user_content},
            ]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            print(f"[Warning] chat template rejected a system role ({e}); "
                  "folding the system prompt into the user turn.")
            messages = [
                {"role": "user", "content": f"{system_prompt}\n\n{raw_user_content}"},
            ]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return self._log_first_prompt(formatted)

    def _log_first_prompt(self, formatted_prompt):
        """Print the exact templated prompt once.

        Lets a smoke run confirm the wrapper the model actually receives (e.g.
        that LightGPT/Llama2 gets the format it was fine-tuned for) -- the
        decision log stores the raw prompt, not this final templated string.
        """
        if not self._logged_first_prompt:
            print(
                "[Info] First formatted prompt sent to the model:\n"
                "-------- BEGIN FORMATTED PROMPT --------\n"
                f"{formatted_prompt}\n"
                "-------- END FORMATTED PROMPT --------"
            )
            self._logged_first_prompt = True
        return formatted_prompt

    def inference(self, raw_prompt):
        formatted_prompt = self._format_prompt(raw_prompt)
        inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=LLM_MAX_NEW_TOKENS,
                temperature=LLM_TEMPERATURE,
                do_sample=LLM_DO_SAMPLE
            )

        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0, input_length:]
        self.last_usage = {
            "prompt_tokens": int(input_length),
            "completion_tokens": int(generated_tokens.shape[0]),
        }
        self.last_formatted_prompt = formatted_prompt
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)