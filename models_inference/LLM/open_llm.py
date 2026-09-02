import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from configurations import (
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    LLM_DO_SAMPLE,
    LLM_SYSTEM_PROMPT,
)


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# The chat-template variable a hybrid model exposes to switch thinking on or
# off. Different families spell it differently; initialize_llm picks whichever
# one the loaded template actually references.
THINKING_TEMPLATE_VARS = ("enable_thinking", "thinking")


def split_reasoning(text):
    """Split a completion into (reasoning, answer).

    A hybrid model emits its chain of thought inline as <think>...</think>
    before the answer. That block must not reach the signal parser -- reasoning
    text names phases constantly, so a regex over it would pick a phase the
    model rejected. Returns (None, text) when there is no block.
    """
    if THINK_CLOSE in text:
        reasoning, _, answer = text.partition(THINK_CLOSE)
        # Some templates pre-open <think> in the prompt, so only the closing
        # tag comes back from the model.
        reasoning = reasoning.split(THINK_OPEN, 1)[-1]
        return reasoning.strip(), answer.strip()
    if THINK_OPEN in text:
        # Truncated: the budget ran out mid-thought. The empty answer becomes a
        # parse error downstream, which holds the phase -- the honest outcome.
        answer, _, reasoning = text.partition(THINK_OPEN)
        return reasoning.strip(), answer.strip()
    return None, text


class LLM_Inference:
    def __init__(self, llm_path, max_new_tokens=None, reasoning="auto"):
        self.llm_path_arg = llm_path
        self.llm_path = self._resolve_snapshot_path(llm_path)
        self.reasoning = reasoning
        # Set in initialize_llm once the real chat template is known.
        self.thinking_var = None
        if max_new_tokens == 0:
            raise ValueError(
                "--max_new_tokens 0 (uncapped) is an OpenRouter-only setting: "
                "HuggingFace generate() needs a finite budget. Pass a real "
                "number for a local model.")
        self.max_new_tokens = max_new_tokens or LLM_MAX_NEW_TOKENS
        self.model = None
        self.tokenizer = None
        self.model_family = None
        self._logged_first_prompt = False
        # Set by every inference() call, read by the runner's recorder.
        self.last_usage = None
        # Per-sequence token counts from the most recent inference_batch().
        self.last_usage_batch = None
        self.last_formatted_prompt = None
        # Chain of thought for the most recent call, mirroring the OpenRouter
        # backend so the runner's recorder reads both the same way.
        self.last_reasoning = None
        self.last_reasoning_batch = None

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

    def initialize_llm(self, torch_dtype=torch.float16, device_map="auto"):
        if torch.cuda.is_available():
            print(f"[Info] CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("[Warning] CUDA not available, running on CPU")

        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path, trust_remote_code=True)
        # Batched generation needs left-padding for a causal LM (real tokens
        # right-aligned so generation continues correctly) and a pad token.
        # Harmless to the single-prompt inference() path, which never pads.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )
        device_map = getattr(self.model, "hf_device_map", None)
        if device_map is not None:
            print(f"[Info] Model device map: {device_map}")
        else:
            print(f"[Info] No hf_device_map; model on: {self.model.device}")
        self.model_family = self._detect_model_family()
        print(f"[Info] Auto-detected model family: {self.model_family}")
        self.thinking_var = self._resolve_thinking_var()

    def _resolve_thinking_var(self):
        """Which template variable --reasoning should set, or None for 'auto'.

        apply_chat_template hands unknown kwargs to Jinja, where an
        unreferenced variable is silently ignored -- so a model whose template
        has no thinking switch would accept --reasoning off and think anyway.
        Fail loudly here instead.
        """
        if self.reasoning == "auto":
            return None
        template = getattr(self.tokenizer, "chat_template", "") or ""
        for name in THINKING_TEMPLATE_VARS:
            if name in template:
                print(f"[Info] Reasoning {self.reasoning}: setting "
                      f"{name}={self.reasoning == 'on'} on the chat template.")
                return name
        raise ValueError(
            f"--reasoning {self.reasoning} was requested, but the chat template of "
            f"{self.llm_path} references none of {list(THINKING_TEMPLATE_VARS)}, so "
            "the setting would be silently ignored. This model has no thinking "
            "switch -- run it with --reasoning auto.")

    def _template_kwargs(self):
        if not self.thinking_var:
            return {}
        return {self.thinking_var: self.reasoning == "on"}

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
                "max_new_tokens": self.max_new_tokens,
                "temperature": LLM_TEMPERATURE,
                "do_sample": LLM_DO_SAMPLE,
                "reasoning": self.reasoning,
                "thinking_template_var": self.thinking_var,
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
                messages, tokenize=False, add_generation_prompt=True,
                **self._template_kwargs()
            )
        except Exception as e:
            print(f"[Warning] chat template rejected a system role ({e}); "
                  "folding the system prompt into the user turn.")
            messages = [
                {"role": "user", "content": f"{system_prompt}\n\n{raw_user_content}"},
            ]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **self._template_kwargs()
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
                max_new_tokens=self.max_new_tokens,
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
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        reasoning, answer = split_reasoning(text)
        self.last_reasoning = reasoning
        return answer

    def inference_batch(self, raw_prompts):
        """Batched sibling of inference(): one generate() over many prompts.

        With greedy decoding (LLM_DO_SAMPLE=False), left-padding + the attention
        mask make every prompt generate as if it were run alone -- the outputs
        are the single-call outputs, modulo GPU float-point reduction order
        (verified by tests/verify_batch_equivalence.py). Returns decoded
        completions aligned to raw_prompts, and fills last_usage_batch
        (per-sequence token counts) + last_formatted_prompt so the recorder can
        attribute usage exactly as the single path does.

        Notes / assumptions:
          - Equivalence is only claimed under greedy decoding; with sampling,
            one shared RNG stream would order draws differently than N calls.
          - Correct position ids under left-padding rely on the model deriving
            them from the attention mask (Qwen2 and every HF-standard causal LM
            do this); an exotic trust_remote_code model that ignores the mask
            would need explicit position_ids.
        """
        if not raw_prompts:
            self.last_usage_batch = []
            self.last_reasoning_batch = []
            return []

        formatted = [self._format_prompt(p) for p in raw_prompts]
        # padding=True left-pads to the longest sequence (padding_side set in
        # initialize_llm); the attention mask masks the pad tokens out entirely.
        inputs = self.tokenizer(
            formatted, return_tensors="pt", padding=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=LLM_TEMPERATURE,
                do_sample=LLM_DO_SAMPLE,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Left-padding makes every real prompt end at the same column, so the
        # generated tokens start at the shared padded input_length for all rows.
        input_length = inputs.input_ids.shape[1]
        stop_ids = self._stop_token_ids()
        texts, usage, reasonings = [], [], []
        for i in range(len(formatted)):
            generated_tokens = outputs[i, input_length:]
            reasoning, answer = split_reasoning(
                self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            )
            texts.append(answer)
            reasonings.append(reasoning)
            usage.append({
                "prompt_tokens": int(inputs.attention_mask[i].sum()),
                "completion_tokens": self._completion_length(generated_tokens, stop_ids),
            })
        self.last_usage_batch = usage
        self.last_reasoning_batch = reasonings
        self.last_formatted_prompt = formatted[0]
        return texts

    def _stop_token_ids(self):
        """The token ids that end generation. Used to count batched completion
        lengths the way the single path does -- up to and including the stop
        token -- instead of counting non-pad tokens, which would drop the
        terminal EOS whenever pad_token was aliased to eos (see initialize_llm)."""
        stop = getattr(self.model.generation_config, "eos_token_id", None)
        if stop is None:
            stop = self.tokenizer.eos_token_id
        if isinstance(stop, int):
            stop = [stop]
        return set(stop or [])

    @staticmethod
    def _completion_length(generated_tokens, stop_ids):
        """Tokens the model actually produced for one row: up to and including
        the first stop token (matching the single path's generated length),
        excluding the right-padding that fills finished rows to the batch width.
        Robust to pad_token_id == eos_token_id."""
        for position, token in enumerate(generated_tokens.tolist()):
            if token in stop_ids:
                return position + 1
        return int(generated_tokens.shape[0])