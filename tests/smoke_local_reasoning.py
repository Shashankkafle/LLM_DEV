"""Offline checks for the local backend's reasoning toggle (--reasoning).

Covers the two halves that a real GPU run cannot cheaply re-check: that the
chat-template kwarg is actually sent (and refused when the template has no
thinking switch), and that an inline <think> block is split off the answer
before the signal parser ever sees it.

Run: python tests/smoke_local_reasoning.py
"""

import sys

sys.path.insert(0, ".")

from models_inference.LLM.open_llm import LLM_Inference, split_reasoning

failures = []


def check(cond, msg):
    print(f"[{'ok  ' if cond else 'FAIL'}] {msg}")
    if not cond:
        failures.append(msg)


class FakeTokenizer:
    def __init__(self, chat_template):
        self.chat_template = chat_template
        self.last_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.last_kwargs = kwargs
        return "PROMPT"


def make_llm(reasoning, chat_template):
    """An LLM_Inference wired to a fake tokenizer, skipping initialize_llm (no
    model weights, no GPU). Mirrors what initialize_llm sets by hand."""
    llm = LLM_Inference("fake/model", max_new_tokens=64, reasoning=reasoning)
    llm.tokenizer = FakeTokenizer(chat_template)
    llm.model_family = "chatml"
    llm.thinking_var = llm._resolve_thinking_var()
    return llm


QWEN_TEMPLATE = "<|im_start|>{% if enable_thinking %}<think>{% endif %}"
PLAIN_TEMPLATE = "<|im_start|>system"

# --- 1. the template kwarg -----------------------------------------------

llm = make_llm("auto", QWEN_TEMPLATE)
llm._format_prompt("hi")
check("enable_thinking" not in llm.tokenizer.last_kwargs,
      "--reasoning auto sends no thinking kwarg (existing runs stay identical)")

llm = make_llm("on", QWEN_TEMPLATE)
llm._format_prompt("hi")
check(llm.tokenizer.last_kwargs.get("enable_thinking") is True,
      "--reasoning on sets enable_thinking=True")

llm = make_llm("off", QWEN_TEMPLATE)
llm._format_prompt("hi")
check(llm.tokenizer.last_kwargs.get("enable_thinking") is False,
      "--reasoning off sets enable_thinking=False")

llm = make_llm("off", "{% if thinking %}<think>{% endif %}")
llm._format_prompt("hi")
check(llm.tokenizer.last_kwargs.get("thinking") is False,
      "a template spelling it 'thinking' gets that kwarg instead")

try:
    make_llm("off", PLAIN_TEMPLATE)
    check(False, "a template with no thinking switch must refuse --reasoning off")
except ValueError:
    check(True, "a template with no thinking switch refuses --reasoning off")

check(make_llm("auto", PLAIN_TEMPLATE).thinking_var is None,
      "...but --reasoning auto still works on such a model")

# --- 2. splitting the chain of thought ------------------------------------

reasoning, answer = split_reasoning("<think>weighing NS</think><signal>ETWT</signal>")
check((reasoning, answer) == ("weighing NS", "<signal>ETWT</signal>"),
      "a full <think> block is split off the answer")

reasoning, answer = split_reasoning("weighing NS</think><signal>ETWT</signal>")
check((reasoning, answer) == ("weighing NS", "<signal>ETWT</signal>"),
      "a pre-opened block (only the closing tag returned) splits too")

check(split_reasoning("<signal>ETWT</signal>") == (None, "<signal>ETWT</signal>"),
      "a non-thinking completion is returned untouched, with no reasoning")

reasoning, answer = split_reasoning("<think>ran out of budget mid-thought")
check(answer == "" and reasoning == "ran out of budget mid-thought",
      "a truncated block leaves an empty answer (a parse error, i.e. hold)")

import runner  # noqa: E402  (imported late: pulls in SUMO-side config)

thinking_names_wrong_phase = (
    "<think>maybe <signal>ETWT</signal></think><signal>NTST</signal>")
check(runner.parse_llm_signal(split_reasoning(thinking_names_wrong_phase)[1]) == "NTST",
      "the parser sees only the answer, not a phase named inside the reasoning")
check(runner.parse_llm_signal(thinking_names_wrong_phase) == "ETWT",
      "...which it would otherwise get wrong (this is why the split exists)")

print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED")
    for msg in failures:
        print(f"  - {msg}")
    sys.exit(1)
print("ALL CHECKS PASSED")
