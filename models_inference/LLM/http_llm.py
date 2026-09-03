"""HTTP-served LLM inference: one backend for every OpenAI-compatible server.

The backend is chosen by a scheme prefix on --llm_path, so trying a new model
is a different --llm_path value rather than new code:

    vllm:qwen2.5_14b                  a model served by a vLLM server
    openrouter:google/gemma-3-27b-it  a model hosted by OpenRouter

Both speak /chat/completions over stdlib urllib, which keeps the backend
dependency-free. The server owns the chat template, the serving precision and
the device placement, so none of that lives here any more.

Exposes the duck-typed surface the runner consumes: initialize_llm(),
inference(), inference_batch(), describe(), and the last_usage /
last_usage_batch / last_formatted_prompt / last_reasoning /
last_reasoning_batch side-channels.
"""

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from configurations import (
    LLM_MAX_NEW_TOKENS,
    LLM_REQUEST_TIMEOUT_S,
    LLM_TEMPERATURE,
    LLM_SYSTEM_PROMPT,
)

MAX_ATTEMPTS = 4
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# Markers that mean a chain of thought survived _split and is about to reach the
# signal parser. Each belongs to a family whose delimiters a server only strips
# when it was launched with the matching --reasoning-parser.
REASONING_MARKERS = (THINK_OPEN, "<|channel>", "<|start|>assistant")


class _RetryableError(Exception):
    """A transient failure worth another attempt (rate limit, 5xx, network)."""


def split_reasoning(text):
    """Split a completion into (reasoning, answer).

    A thinking model emits its chain of thought inline as <think>...</think>
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


def _as_text(messages):
    """The messages as a single string, for the manifest's record of exactly
    what the model was sent."""
    return json.dumps(messages, indent=2)


def _send(request, timeout_s):
    """One request. Transient failures raise _RetryableError; everything else is
    fatal and raises straight through."""
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        if exc.code in RETRYABLE_STATUS:
            raise _RetryableError(f"HTTP {exc.code}: {detail}") from exc
        raise RuntimeError(f"HTTP {exc.code} from {request.full_url}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise _RetryableError(repr(exc)) from exc

    # Some servers report upstream failures in an HTTP-200 body.
    if isinstance(body, dict) and body.get("error"):
        raise _RetryableError(f"error in response body: {body['error']}")
    return body


def _extract_content(body):
    """The assistant's text. An empty completion (e.g. the model spent its whole
    budget on reasoning) is returned as "" so the runner classifies it as a parse
    error, exactly as a truncated generation behaves -- an infrastructure
    failure must never be logged as a model failure, so those raise instead."""
    return _message_of(body, required=True).get("content") or ""


def _message_of(body, required=False):
    try:
        return body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        if required:
            raise RuntimeError(f"Unexpected response shape: {body}") from exc
        return {}


def _extract_usage(body):
    """Token counts under the same keys the recorder already expects.

    reasoning_tokens is a subset of completion_tokens, not an addition to it --
    it is billed as output either way, and is broken out so a thinking model's
    spend is attributable.

    finish_reason rides along here because it is the only thing that separates
    "the model had nothing useful to say" from "the budget ran out mid-answer":
    a thinking model spends reasoning tokens out of max_tokens, so an
    undersized cap yields finish_reason == "length" with empty content, which
    is indistinguishable from a genuine parse failure downstream."""
    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    try:
        finish_reason = body["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError):
        finish_reason = None
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "finish_reason": finish_reason,
    }


class HTTPChatLLM:
    """Everything the two servers share. Subclasses differ in four places:
    whether an API key is required, what preflight proves, what extra keys a
    request carries, and how reasoning is separated from the answer."""

    BACKEND = "http"
    DEFAULT_BASE_URL = None
    BASE_URL_ENV = None
    API_KEY_ENV = None
    API_KEY_REQUIRED = False

    def __init__(self, llm_path, max_new_tokens=None, timeout_s=None,
                 reasoning_max_tokens=None, reasoning="auto", quantization="none"):
        self.llm_path_arg = llm_path
        self.model = _model_from_path(llm_path)
        # Explicit None check, not `or`: 0 is the "uncapped" sentinel and must
        # not fall back to the default.
        self.max_new_tokens = (LLM_MAX_NEW_TOKENS if max_new_tokens is None
                               else max_new_tokens)
        # Bounds the thinking specifically, leaving the rest of max_tokens for
        # the answer. A thinking model's cost is bimodal -- a typical decision
        # is cheap, a few run away and eat the whole budget -- so capping the
        # total trades those runaways for truncation, while capping the
        # reasoning cuts them off with room left to answer.
        self.reasoning_max_tokens = reasoning_max_tokens
        self.reasoning = reasoning
        # Never sent anywhere: the server owns its precision. Recorded so an
        # AWQ-served run never pools with an fp16-served one. See describe().
        self.quantization = quantization
        # A chain-of-thought model can think for minutes on one decision; the
        # 120 s default is sized for the non-thinking arms, and a timeout is
        # retryable, so an undersized value burns MAX_ATTEMPTS generations.
        self.timeout_s = timeout_s or LLM_REQUEST_TIMEOUT_S
        self.base_url = os.environ.get(
            self.BASE_URL_ENV, self.DEFAULT_BASE_URL).rstrip("/")
        self.api_key = None
        # What the server actually answered with. A served run is not
        # reproducible the way an in-process one was, so the manifest should
        # name the real server.
        self.resolved_model = None
        self.resolved_provider = None
        self.server_info = {}
        self._logged_first_prompt = False
        self._budget_exhausted_calls = 0
        self._leaked_reasoning_calls = 0
        # Set by every inference() call, read by the runner's recorder.
        self.last_usage = None
        # Per-prompt values from the most recent inference_batch().
        self.last_usage_batch = None
        self.last_reasoning = None
        self.last_reasoning_batch = None
        self.last_formatted_prompt = None

    # -- lifecycle ---------------------------------------------------------

    def initialize_llm(self):
        self.api_key = os.environ.get(self.API_KEY_ENV) if self.API_KEY_ENV else None
        if self.API_KEY_REQUIRED and not self.api_key:
            raise RuntimeError(
                f"{self.API_KEY_ENV} is not set, but --llm_path "
                f"{self.llm_path_arg!r} needs it.")
        self._preflight()
        print(f"[Info] {self.BACKEND} backend ready: model={self.model} "
              f"base_url={self.base_url} "
              f"max_tokens={self.max_new_tokens or 'uncapped'}")
        # Uncapped, a single runaway generation can outlast any fixed timeout --
        # and a timeout is retryable, so it would be billed MAX_ATTEMPTS times
        # and still fail. Nothing bounds a decision's cost but the clock.
        if not self.max_new_tokens and self.timeout_s < 600:
            print(f"[Warning] max_tokens is uncapped but --request_timeout is "
                  f"{self.timeout_s}s. A long generation will time out, and a "
                  f"timeout is retried up to {MAX_ATTEMPTS} times -- you would "
                  f"pay for every attempt. Raise --request_timeout.")

    def _preflight(self):
        """Prove the server can serve this model before SUMO starts.

        The runner treats a failed inference as "hold the current phase", so a
        persistent fault would otherwise burn a whole run producing silently
        degraded control instead of crashing."""
        raise NotImplementedError

    def describe(self):
        """Model + generation config for the run manifest. Never includes the
        API key. Call after initialize_llm(), which fills the server details."""
        return {
            "llm_path_arg": self.llm_path_arg,
            "backend": self.BACKEND,
            "model": self.model,
            "resolved_model": self.resolved_model,
            "resolved_provider": self.resolved_provider,
            "base_url": self.base_url,
            "server": self.server_info or None,
            "generation": {
                "max_new_tokens": self.max_new_tokens,
                "reasoning_max_tokens": self.reasoning_max_tokens,
                "temperature": LLM_TEMPERATURE,
                "reasoning": self.reasoning,
                # The only client-side record of what --reasoning did, now that
                # the chat template is invisible from here.
                "extra_payload_sent": self._extra_payload() or None,
            },
            # Operator-asserted and unverifiable over the API: it says how the
            # server was launched, and exists so identity keeps precisions apart.
            "quantization_declared": self.quantization,
            # A served run's reproducibility lives in the launch command, which
            # is outside this repo. Recorded when serve_vllm.sh exported it.
            "serve_cmd": os.environ.get("VLLM_SERVE_CMD"),
            "request_timeout_s": self.timeout_s,
            "system_prompt": LLM_SYSTEM_PROMPT,
        }

    # -- prompts -----------------------------------------------------------

    def _format_prompt(self, raw_user_content):
        """The chat messages for one decision. There is no chat template to
        apply here -- the server owns that."""
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

    # -- transport ---------------------------------------------------------

    def _server_root(self):
        """The server's root, for endpoints that sit outside /v1."""
        if self.base_url.endswith("/v1"):
            return self.base_url[: -len("/v1")]
        return self.base_url

    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(self, url, payload=None):
        """One request, retrying transient failures with exponential backoff.
        payload=None sends a GET. Returns the decoded response body."""
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
            headers=self._headers(),
        )
        last_error = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                body = _send(request, self.timeout_s)
                self._remember_routing(body)
                return body
            except _RetryableError as exc:
                last_error = exc
                if attempt < MAX_ATTEMPTS - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(
            f"{self.BACKEND} request to {url} failed after "
            f"{MAX_ATTEMPTS} attempts: {last_error}")

    def _post_json(self, payload):
        return self._request(f"{self.base_url}/chat/completions", payload)

    def _remember_routing(self, body):
        if not isinstance(body, dict):
            return
        self.resolved_model = body.get("model") or self.resolved_model
        self.resolved_provider = body.get("provider") or self.resolved_provider

    # -- generation --------------------------------------------------------

    def _extra_payload(self):
        """Scheme-specific request keys. Empty means the request is exactly the
        plain OpenAI shape."""
        return {}

    def _complete(self, messages):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": LLM_TEMPERATURE,
        }
        # max_new_tokens=0 means uncapped: omit max_tokens entirely so the model
        # runs to its own stop or its context limit, rather than to a number we
        # picked. Sending 0 would be a request for zero tokens.
        if self.max_new_tokens:
            payload["max_tokens"] = self.max_new_tokens
        payload.update(self._extra_payload())
        return self._post_json(payload)

    def _split(self, content, body):
        """Separate one response into (reasoning, answer)."""
        raise NotImplementedError

    def _warn_if_budget_exhausted(self, content, usage):
        """Say so, loudly, when the cap -- not the model -- ate the answer.

        Downstream this is just a parse error that holds the current phase, so
        without this a whole run can degrade to "hold everything" while still
        producing a complete, plausible-looking results file."""
        if content or usage.get("finish_reason") != "length":
            return
        self._budget_exhausted_calls += 1
        if self._budget_exhausted_calls <= 3:
            print(f"[Warning] Empty completion at max_tokens={self.max_new_tokens} "
                  f"(finish_reason=length, reasoning_tokens="
                  f"{usage.get('reasoning_tokens')}). The model spent its whole "
                  f"budget thinking; raise --max_new_tokens.")

    def _resolve(self, body):
        """One response body -> (answer, usage, reasoning)."""
        usage = _extract_usage(body)
        reasoning, answer = self._split(_extract_content(body), body)
        self._warn_if_budget_exhausted(answer, usage)
        return answer, usage, reasoning

    def inference(self, raw_prompt):
        messages = self._format_prompt(raw_prompt)
        body = self._complete(messages)
        answer, self.last_usage, self.last_reasoning = self._resolve(body)
        self.last_formatted_prompt = _as_text(messages)
        return answer

    def inference_batch(self, raw_prompts):
        """Concurrent sibling of inference(): one independent request per prompt.

        There is no shared generate() call, so the outputs are the single-call
        outputs by construction -- no padding or decoding subtleties to verify.
        A vLLM server fuses these into one continuous batch server-side, which
        is where the throughput comes from. Prompts are formatted on this thread
        so the first-prompt logging and last_formatted_prompt stay race-free.

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

        resolved = [self._resolve(body) for body in bodies]
        self.last_usage_batch = [usage for _, usage, _ in resolved]
        self.last_reasoning_batch = [reasoning for _, _, reasoning in resolved]
        self.last_formatted_prompt = _as_text(batch[0])
        return [answer for answer, _, _ in resolved]


class OpenRouterLLM(HTTPChatLLM):
    """OpenRouter's hosted API. Routes across providers, so a run is not
    reproducible the way a self-hosted one is -- describe() records where it
    actually landed."""

    BACKEND = "openrouter"
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    BASE_URL_ENV = "OPENROUTER_BASE_URL"
    API_KEY_ENV = "OPENROUTER_API_KEY"
    API_KEY_REQUIRED = True

    def _preflight(self):
        """One tiny request. A bad key, an unknown model or a dead network must
        fail here, before SUMO starts."""
        self._post_json({
            "model": self.model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        })

    def _extra_payload(self):
        # Only sent when asked for: a non-thinking model would be handed a
        # parameter it never needs, and the non-CoT arms must stay byte-identical.
        if self.reasoning_max_tokens:
            return {"reasoning": {"max_tokens": self.reasoning_max_tokens}}
        return {}

    def _split(self, content, body):
        """Read the separate reasoning field; never touch the content.

        Load-bearing: this is exactly the pre-migration behavior, so completed
        OpenRouter runs stay comparable to future ones. Do not add the inline
        <think> split here -- that would silently change what the signal parser
        sees for every hosted model that thinks out loud in plain content.
        """
        message = _message_of(body)
        if message.get("reasoning"):
            return message["reasoning"], content
        # Newer providers only populate the structured form.
        details = message.get("reasoning_details") or []
        chunks = [d.get("text") or d.get("summary") or "" for d in details
                  if isinstance(d, dict)]
        return ("".join(chunks).strip() or None), content


class VLLMLLM(HTTPChatLLM):
    """A local vLLM server. --llm_path names the served model
    (`vllm:qwen2.5_14b`), never a URL -- the endpoint lives in VLLM_BASE_URL, so
    moving the server between ports does not invalidate a grid's run identity."""

    BACKEND = "vllm"
    DEFAULT_BASE_URL = "http://localhost:8000/v1"
    BASE_URL_ENV = "VLLM_BASE_URL"
    API_KEY_ENV = "VLLM_API_KEY"
    API_KEY_REQUIRED = False

    THINKING_TEMPLATE_VAR = "enable_thinking"

    def _preflight(self):
        self._check_model_is_served()
        self._record_server_version()
        if self.reasoning != "auto":
            self._check_thinking_var_is_honored()
        if self.reasoning_max_tokens:
            print("[Warning] --reasoning_max_tokens is an OpenRouter-only "
                  "setting; vLLM has no separate reasoning budget. Ignored.")

    def _check_model_is_served(self):
        """Fail with the served names rather than an opaque HTTP 400, and record
        what the server actually loaded -- `root` is the path or repo id behind
        the served name, the replacement for the old resolved_snapshot_path."""
        listed = self._request(f"{self.base_url}/models").get("data") or []
        served = [entry.get("id") for entry in listed]
        if self.model not in served:
            raise RuntimeError(
                f"model {self.model!r} is not served at {self.base_url}; "
                f"this server serves {served}. Pass "
                f"--llm_path vllm:{served[0] if served else '<served-model-name>'}.")
        self.resolved_model = self.model
        self.server_info["model_entry"] = next(
            entry for entry in listed if entry.get("id") == self.model)

    def _record_server_version(self):
        """Provenance must never fail a run: a proxy or another
        OpenAI-compatible server has no /version, which is not an error."""
        try:
            body = self._request(f"{self._server_root()}/version")
            self.server_info["version"] = body.get("version")
        except Exception as exc:
            self.server_info["version"] = None
            print(f"[Warning] No /version at {self._server_root()} ({exc}); "
                  "the server version will not be recorded.")

    def _check_thinking_var_is_honored(self):
        """Prove --reasoning actually reaches the model.

        The retired local backend grepped the chat template and raised when it
        referenced no thinking variable, specifically so --reasoning off could
        not be silently ignored. From here the template is invisible and vLLM
        drops unknown chat_template_kwargs without complaint, so ask /tokenize
        to render the same messages both ways: identical token counts mean the
        template never branches on the variable. Two requests, no generation.
        """
        counts = [self._tokenized_length(value) for value in (True, False)]
        if None in counts:
            print("[Warning] This server has no /tokenize, so --reasoning "
                  f"{self.reasoning} could not be verified. If the model's chat "
                  "template has no thinking switch, the setting is silently "
                  "ignored.")
            return
        if counts[0] == counts[1]:
            raise ValueError(
                f"--reasoning {self.reasoning} was requested, but the chat "
                f"template of {self.model!r} renders identically with "
                f"{self.THINKING_TEMPLATE_VAR} true and false ({counts[0]} "
                "tokens both ways), so the setting would be silently ignored. "
                "This model has no thinking switch -- run it with --reasoning "
                "auto, or control thinking with the server's --reasoning-parser.")

    def _tokenized_length(self, thinking):
        try:
            body = self._request(f"{self._server_root()}/tokenize", {
                "model": self.model,
                "messages": [{"role": "user", "content": "ping"}],
                "chat_template_kwargs": {self.THINKING_TEMPLATE_VAR: thinking},
            })
        except Exception:
            return None
        return body.get("count")

    def _extra_payload(self):
        # 'auto' sends no key at all, so those requests are byte-identical to a
        # run launched without the flag.
        if self.reasoning == "auto":
            return {}
        return {"chat_template_kwargs": {
            self.THINKING_TEMPLATE_VAR: self.reasoning == "on"}}

    def _split(self, content, body):
        """Prefer the server's parsed reasoning; fall back to the inline split.

        vLLM only populates reasoning_content when it was launched with a
        --reasoning-parser. Without one a thinking model returns its chain of
        thought inline, so the <think> split is the safety net. Skipping the
        split when reasoning_content is present also avoids re-splitting an
        answer that legitimately contains the literal closing tag.
        """
        reasoning = _message_of(body).get("reasoning_content")
        if reasoning:
            return reasoning, content
        reasoning, answer = split_reasoning(content)
        self._warn_if_reasoning_leaked(answer)
        return reasoning, answer

    def _warn_if_reasoning_leaked(self, answer):
        """A chain of thought reaching parse_llm_signal picks a phase the model
        rejected, and looks like a perfectly normal decision in the log. Say so
        rather than letting a whole run degrade quietly."""
        marker = next((m for m in REASONING_MARKERS if m in answer), None)
        if marker is None:
            return
        self._leaked_reasoning_calls += 1
        if self._leaked_reasoning_calls <= 3:
            print(f"[Warning] The answer still contains {marker!r} after "
                  "splitting, so reasoning is reaching the signal parser. "
                  "Launch the server with the --reasoning-parser for this "
                  "model family.")


SCHEMES = {
    "openrouter:": OpenRouterLLM,
    "vllm:": VLLMLLM,
}

RETIRED_LOCAL_BACKEND = (
    "--llm_path {path!r} names no backend. The in-process HuggingFace backend "
    "was retired; serve the weights with vLLM instead and pass the served model "
    "name:\n"
    "    vllm serve {path} --served-model-name my_model\n"
    "    python runner.py --llm_path vllm:my_model\n"
    "Set VLLM_BASE_URL if the server is not on http://localhost:8000/v1. "
    "Hosted models still work as 'openrouter:<provider>/<model>'."
)


def strip_scheme(llm_path):
    """'vllm:qwen2.5_14b' -> 'qwen2.5_14b'. Used for the run-directory tag and
    the results' model column, so those stay scheme-agnostic."""
    for prefix in SCHEMES:
        if llm_path.startswith(prefix):
            return llm_path[len(prefix):]
    return llm_path


def _model_from_path(llm_path):
    """The model name a scheme-prefixed path carries. Model ids contain '/' but
    never ':', so splitting on the first ':' is safe."""
    model = llm_path.split(":", 1)[1].strip() if ":" in llm_path else ""
    if not model:
        raise ValueError(
            f"--llm_path {llm_path!r} names no model. Expected "
            f"'<scheme>:<model>', e.g. 'vllm:qwen2.5_14b' or "
            f"'openrouter:google/gemma-3-27b-it'.")
    return model


def build(llm_path, max_new_tokens=None, request_timeout=None,
          reasoning_max_tokens=None, reasoning="auto", quantization="none"):
    """The one backend-selection point in the pipeline. Every scheme satisfies
    the same interface, so nothing downstream knows which server it is talking
    to."""
    for prefix, backend in SCHEMES.items():
        if llm_path.startswith(prefix):
            return backend(llm_path, max_new_tokens=max_new_tokens,
                           timeout_s=request_timeout,
                           reasoning_max_tokens=reasoning_max_tokens,
                           reasoning=reasoning, quantization=quantization)
    raise ValueError(RETIRED_LOCAL_BACKEND.format(path=llm_path))
