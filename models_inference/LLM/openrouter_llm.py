"""OpenRouter-hosted inference, a drop-in alternative to open_llm.LLM_Inference.

Selected by an ``openrouter:<model>`` --llm_path (see runner.build_llm), so a
hosted arm is just a different model string in experiments.py -- no new CLI flag
and no run_matrix plumbing. Speaks OpenRouter's OpenAI-compatible
/chat/completions over stdlib urllib, which keeps the backend dependency-free.

Exposes the same duck-typed surface the runner consumes from LLM_Inference:
initialize_llm(), inference(), inference_batch(), describe(), and the
last_usage / last_usage_batch / last_formatted_prompt side-channels, plus
last_reasoning / last_reasoning_batch for models that return their chain of
thought as separate reasoning tokens.
"""

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from configurations import (
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    LLM_SYSTEM_PROMPT,
)

OPENROUTER_PREFIX = "openrouter:"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"
BASE_URL_ENV = "OPENROUTER_BASE_URL"

REQUEST_TIMEOUT_S = 120
MAX_ATTEMPTS = 4
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class _RetryableError(Exception):
    """A transient failure worth another attempt (rate limit, 5xx, network)."""


def _model_from_path(llm_path):
    """``openrouter:google/gemma-3-27b-it`` -> ``google/gemma-3-27b-it``.
    Model ids contain '/' but never ':', so splitting on the first ':' is safe."""
    model = llm_path.split(":", 1)[1].strip() if ":" in llm_path else ""
    if not model:
        raise ValueError(
            f"--llm_path {llm_path!r} names no model. Expected "
            f"'{OPENROUTER_PREFIX}<provider>/<model>', e.g. "
            f"'{OPENROUTER_PREFIX}google/gemma-3-27b-it'.")
    return model


def _as_text(messages):
    """The messages as a single string, for the manifest's record of exactly
    what the model was sent (the local backend stores a templated string there)."""
    return json.dumps(messages, indent=2)


def _send(request):
    """One POST. Transient failures raise _RetryableError; everything else is
    fatal and raises straight through."""
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        if exc.code in RETRYABLE_STATUS:
            raise _RetryableError(f"HTTP {exc.code}: {detail}") from exc
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise _RetryableError(repr(exc)) from exc

    # OpenRouter reports some upstream failures in an HTTP-200 body.
    if isinstance(body, dict) and body.get("error"):
        raise _RetryableError(f"error in response body: {body['error']}")
    return body


def _extract_content(body):
    """The assistant's text. An empty completion (e.g. the model spent its whole
    budget on reasoning) is returned as "" so the runner classifies it as a parse
    error, exactly as a truncated local generation behaves -- an infrastructure
    failure must never be logged as a model failure, so those raise instead."""
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected OpenRouter response shape: {body}") from exc
    return message.get("content") or ""


def _extract_usage(body):
    """Token counts under the same keys the recorder already expects.

    reasoning_tokens is a subset of completion_tokens, not an addition to it --
    it is billed as output either way, and is broken out so a thinking model's
    spend is attributable."""
    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
    }


def _extract_reasoning(body):
    """The model's chain of thought, when it emits one as separate reasoning
    tokens rather than inside the answer text.

    Models that think out loud in plain content (the Qwen fine-tune, most
    non-thinking models) return None here -- their reasoning is already the
    recorded raw_text. Nothing is requested that the caller did not ask for:
    this only reads what the response happens to carry, so enabling it cannot
    change routing, cost, or the decisions themselves.
    """
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    if message.get("reasoning"):
        return message["reasoning"]
    # Newer providers only populate the structured form.
    details = message.get("reasoning_details") or []
    chunks = [d.get("text") or d.get("summary") or "" for d in details
              if isinstance(d, dict)]
    joined = "".join(chunks).strip()
    return joined or None


class OpenRouter_Inference:
    def __init__(self, llm_path):
        self.llm_path_arg = llm_path
        self.model = _model_from_path(llm_path)
        self.base_url = os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL).rstrip("/")
        self.api_key = None
        # What OpenRouter actually routed to. A hosted run is not reproducible
        # the way a local one is, so the manifest should name the real server.
        self.resolved_model = None
        self.resolved_provider = None
        self._logged_first_prompt = False
        # Set by every inference() call, read by the runner's recorder.
        self.last_usage = None
        # Per-prompt token counts from the most recent inference_batch().
        self.last_usage_batch = None
        # Chain of thought for the most recent call, when the model emits one
        # separately from its answer text. None for non-thinking models.
        self.last_reasoning = None
        self.last_reasoning_batch = None
        self.last_formatted_prompt = None

    def initialize_llm(self):
        self.api_key = os.environ.get(API_KEY_ENV)
        if not self.api_key:
            raise RuntimeError(
                f"{API_KEY_ENV} is not set, but --llm_path "
                f"{self.llm_path_arg!r} needs it.")
        self._preflight()
        print(f"[Info] OpenRouter backend ready: model={self.model} "
              f"resolved={self.resolved_model} provider={self.resolved_provider}")

    def _preflight(self):
        """One tiny request before SUMO starts. A bad key, an unknown model or a
        dead network must fail here: the runner treats a failed inference as
        "hold the current phase", so a persistent fault would otherwise burn a
        whole run producing silently degraded control instead of crashing."""
        self._post_json({
            "model": self.model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        })

    def describe(self):
        """Model + generation config for the run manifest. Never includes the
        API key. Call after initialize_llm(), which fills the resolved routing."""
        return {
            "llm_path_arg": self.llm_path_arg,
            "backend": "openrouter",
            "model": self.model,
            "resolved_model": self.resolved_model,
            "resolved_provider": self.resolved_provider,
            "base_url": self.base_url,
            "generation": {
                "max_new_tokens": LLM_MAX_NEW_TOKENS,
                "temperature": LLM_TEMPERATURE,
            },
            "system_prompt": LLM_SYSTEM_PROMPT,
        }

    def _format_prompt(self, raw_user_content):
        """The chat messages for one decision. There is no chat template to
        apply here -- the server owns that -- so this is just the system + user
        turn the local backend builds before templating."""
        messages = [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": raw_user_content},
        ]
        self._log_first_prompt(_as_text(messages))
        return messages

    def _log_first_prompt(self, formatted_prompt):
        """Print the exact payload once, so a smoke run can confirm what the
        model actually receives -- the decision log stores the raw user prompt,
        not this wrapped form."""
        if not self._logged_first_prompt:
            print(
                "[Info] First formatted prompt sent to the model:\n"
                "-------- BEGIN FORMATTED PROMPT --------\n"
                f"{formatted_prompt}\n"
                "-------- END FORMATTED PROMPT --------"
            )
            self._logged_first_prompt = True
        return formatted_prompt

    def _post_json(self, payload):
        """POST one payload, retrying transient failures with exponential
        backoff. Returns the decoded response body."""
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        last_error = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                body = _send(request)
                self._remember_routing(body)
                return body
            except _RetryableError as exc:
                last_error = exc
                if attempt < MAX_ATTEMPTS - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(
            f"OpenRouter request failed after {MAX_ATTEMPTS} attempts: {last_error}")

    def _remember_routing(self, body):
        self.resolved_model = body.get("model") or self.resolved_model
        self.resolved_provider = body.get("provider") or self.resolved_provider

    def _complete(self, messages):
        return self._post_json({
            "model": self.model,
            "messages": messages,
            "max_tokens": LLM_MAX_NEW_TOKENS,
            "temperature": LLM_TEMPERATURE,
        })

    def inference(self, raw_prompt):
        messages = self._format_prompt(raw_prompt)
        body = self._complete(messages)
        self.last_usage = _extract_usage(body)
        self.last_reasoning = _extract_reasoning(body)
        self.last_formatted_prompt = _as_text(messages)
        return _extract_content(body)

    def inference_batch(self, raw_prompts):
        """Concurrent sibling of inference(): one independent request per prompt.

        Unlike the local backend there is no shared generate() call, so the
        outputs are the single-call outputs by construction -- no padding or
        decoding subtleties to verify. Prompts are formatted on this thread so
        the first-prompt logging and last_formatted_prompt stay race-free.

        A prompt that still fails after its retries propagates: the runner's
        infer_chunk then falls back to per-prompt, isolating the one genuinely
        failing intersection rather than failing the whole cohort.
        """
        if not raw_prompts:
            self.last_usage_batch = []
            self.last_reasoning_batch = []
            return []

        batch = [self._format_prompt(prompt) for prompt in raw_prompts]
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            bodies = list(pool.map(self._complete, batch))

        self.last_usage_batch = [_extract_usage(body) for body in bodies]
        self.last_reasoning_batch = [_extract_reasoning(body) for body in bodies]
        self.last_formatted_prompt = _as_text(batch[0])
        return [_extract_content(body) for body in bodies]
