"""Offline checks for the OpenRouter backend (models_inference/LLM/http_llm).

Runs the real client against a local stand-in server via OPENROUTER_BASE_URL, so
the request shape, retry policy, batch ordering and error taxonomy are all
exercised without a network call, an API key, or a cent of credit.

The error taxonomy is the load-bearing part: the runner treats an exception as
"inference_error" (infrastructure) and an unusable string as
"fallback_parse_error" (the model), and those must not be confused.

Run: python tests/smoke_openrouter_backend.py
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, ".")

failures = []


def check(cond, msg):
    print(f"[{'ok  ' if cond else 'FAIL'}] {msg}")
    if not cond:
        failures.append(msg)


# --- a scriptable stand-in for OpenRouter ------------------------------------

class FakeOpenRouter:
    """Replies with whatever the current script says. Each entry is
    (status, body); the last entry repeats once the script is exhausted."""

    def __init__(self):
        self.script = [(200, self._completion("<signal>ETWT</signal>"))]
        self.requests = []
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}/api/v1"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @staticmethod
    def _completion(content, prompt_tokens=759, completion_tokens=42,
                    reasoning_tokens=None, finish_reason="stop"):
        usage = {"prompt_tokens": prompt_tokens,
                 "completion_tokens": completion_tokens}
        if reasoning_tokens is not None:
            usage["completion_tokens_details"] = {
                "reasoning_tokens": reasoning_tokens}
        return {
            "model": "google/gemma-3-27b-it",
            "provider": "Google AI Studio",
            "choices": [{"message": {"role": "assistant", "content": content},
                         "finish_reason": finish_reason}],
            "usage": usage,
        }

    def _next(self, payload):
        with self._lock:
            self.requests.append(payload)
            return self.script[0] if len(self.script) == 1 else self.script.pop(0)

    def _handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                payload["_auth"] = self.headers.get("Authorization")
                status, body = outer._next(payload)
                encoded = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *args):
                pass

        return Handler

    def shutdown(self):
        self._server.shutdown()


fake = FakeOpenRouter()
os.environ["OPENROUTER_BASE_URL"] = fake.base_url

from models_inference.LLM import http_llm as mod  # noqa: E402
from models_inference.LLM.http_llm import OpenRouterLLM  # noqa: E402
import runner  # noqa: E402

MODEL = "openrouter:google/gemma-3-27b-it"

# A real key must never enter this test: the stand-in server would receive it as
# a Bearer header and keep it in its request log. Stash whatever the operator has
# set, run entirely on a dummy, and put the original back on the way out.
REAL_KEY = os.environ.pop("OPENROUTER_API_KEY", None)
TEST_KEY = "test-key-not-a-real-secret"


def restore_real_key():
    if REAL_KEY is None:
        os.environ.pop("OPENROUTER_API_KEY", None)
    else:
        os.environ["OPENROUTER_API_KEY"] = REAL_KEY


def fresh(script=None, **kwargs):
    """A ready-to-use client, with the fake reset to `script`."""
    fake.script = script or [(200, fake._completion("<signal>ETWT</signal>"))]
    fake.requests.clear()
    llm = OpenRouterLLM(MODEL, **kwargs)
    llm.initialize_llm()
    fake.requests.clear()  # drop the preflight
    return llm


# --- 1. the runner's factory picks this backend up ---------------------------

check(isinstance(runner.build_llm(MODEL), OpenRouterLLM),
      "runner.build_llm routes an openrouter: path to this backend")
try:
    runner.build_llm("models/LLMs/whatever")
    check(False, "an unprefixed path is rejected now the local backend is retired")
except ValueError as exc:
    check("vllm serve" in str(exc),
          "an unprefixed path is rejected now the local backend is retired")

# --- 2. a missing API key stops the run immediately --------------------------
# (the key is already unset -- see REAL_KEY above)

try:
    OpenRouterLLM(MODEL).initialize_llm()
    check(False, "a missing OPENROUTER_API_KEY aborts in initialize_llm()")
except RuntimeError as exc:
    check("OPENROUTER_API_KEY" in str(exc),
          "a missing OPENROUTER_API_KEY aborts in initialize_llm()")
os.environ["OPENROUTER_API_KEY"] = TEST_KEY

# --- 3. a malformed --llm_path is rejected at construction -------------------

try:
    OpenRouterLLM("openrouter:")
    check(False, "an empty model in --llm_path is rejected")
except ValueError:
    check(True, "an empty model in --llm_path is rejected")

# --- 4. preflight really talks to the server before SUMO starts --------------

fake.script = [(200, fake._completion("pong"))]
fake.requests.clear()
llm = OpenRouterLLM(MODEL)
llm.initialize_llm()
check(len(fake.requests) == 1, "initialize_llm() preflights with one request")
check(fake.requests[0]["max_tokens"] == 1,
      "the preflight is a 1-token request, not a full generation")
check(llm.resolved_model == "google/gemma-3-27b-it"
      and llm.resolved_provider == "Google AI Studio",
      "preflight records the resolved model/provider for the manifest")

# --- 5. request shape --------------------------------------------------------

llm = fresh()
output = llm.inference("STATE HERE")
sent = fake.requests[0]
check(sent["model"] == "google/gemma-3-27b-it", "request names the bare model id")
check(sent["_auth"] == f"Bearer {TEST_KEY}", "request carries the bearer token")
check(sent["max_tokens"] == mod.LLM_MAX_NEW_TOKENS
      and sent["temperature"] == mod.LLM_TEMPERATURE,
      "generation settings come from configurations, read at call time")
roles = [m["role"] for m in sent["messages"]]
check(roles == ["system", "user"], "messages are a system + user turn")
check(sent["messages"][0]["content"] == mod.LLM_SYSTEM_PROMPT,
      "the system turn is the shared LLM_SYSTEM_PROMPT")
check(sent["messages"][1]["content"] == "STATE HERE",
      "the user turn is the prompt verbatim, unwrapped")
check(output == "<signal>ETWT</signal>", "inference() returns the assistant text")
check({k: llm.last_usage.get(k) for k in
       ("prompt_tokens", "completion_tokens", "reasoning_tokens", "finish_reason")}
      == {"prompt_tokens": 759, "completion_tokens": 42,
          "reasoning_tokens": None, "finish_reason": "stop"},
      "last_usage carries the server's token counts")
check(runner.parse_llm_signal(output) == "ETWT",
      "the runner's existing parser reads the hosted output unchanged")

# --- 6. describe() is manifest-ready and leaks no secret ---------------------

described = llm.describe()
check(described["backend"] == "openrouter" and described["llm_path_arg"] == MODEL,
      "describe() identifies the backend and the original --llm_path")
check(described["resolved_provider"] == "Google AI Studio",
      "describe() reports which provider actually served the run")
check(TEST_KEY not in json.dumps(described),
      "describe() never exposes the API key")

# --- 6b. a thinking model's budget overrides ---------------------------------
# A CoT model spends reasoning tokens out of max_tokens, so the cap and the
# timeout have to be raisable per run without editing configurations.py.

llm = fresh(max_new_tokens=32000, timeout_s=600)
llm.inference("STATE HERE")
check(fake.requests[0]["max_tokens"] == 32000,
      "an overridden max_new_tokens reaches the wire")
check(llm.describe()["generation"]["max_new_tokens"] == 32000
      and llm.describe()["request_timeout_s"] == 600,
      "describe() records the overrides, so the manifest pins the real budget")
check(fresh().max_new_tokens == mod.LLM_MAX_NEW_TOKENS
      and fresh().timeout_s == mod.LLM_REQUEST_TIMEOUT_S,
      "omitting the overrides keeps the configurations.py defaults")

# reasoning_tokens is broken out of completion_tokens, and finish_reason is what
# tells a budget exhaustion apart from a model that had nothing to say.
llm = fresh([(200, fake._completion("", completion_tokens=1024,
                                    reasoning_tokens=1024,
                                    finish_reason="length"))])
check(llm.inference("STATE HERE") == "",
      "a budget-exhausted call still returns '' rather than raising")
check(llm.last_usage["reasoning_tokens"] == 1024
      and llm.last_usage["finish_reason"] == "length",
      "usage attributes the spend to reasoning and flags the truncation")
fake.script = [(200, fake._completion("<signal>ETWT</signal>"))]

# --- 7. batching: order preserved, usage aligned -----------------------------

llm = fresh()
prompts = [f"PROMPT {i}" for i in range(8)]
fake.script = [(200, fake._completion(f"<signal>ETWT</signal> for {p}",
                                      completion_tokens=i))
               for i, p in enumerate(prompts)]
outputs = llm.inference_batch(prompts)
check(len(outputs) == len(prompts) and len(llm.last_usage_batch) == len(prompts),
      "inference_batch returns one output and one usage per prompt")
sent_order = [r["messages"][1]["content"] for r in fake.requests]
check(sorted(sent_order) == sorted(prompts) and len(fake.requests) == len(prompts),
      "every prompt is sent exactly once")
check(llm.inference_batch([]) == [] and llm.last_usage_batch == [],
      "an empty batch short-circuits without a request")

# Ordering is the contract runner.infer_chunk zips on. Reply with an answer
# derived from each prompt, so misalignment shows up no matter what order the
# concurrent requests happen to complete in.
llm = fresh()
answers = {p: f"answer-{i}" for i, p in enumerate(prompts)}
original_next = fake._next
fake._next = lambda payload: (
    200, fake._completion(answers[payload["messages"][1]["content"]]))
outputs = llm.inference_batch(prompts)
fake._next = original_next
check(outputs == [answers[p] for p in prompts],
      "batch outputs stay aligned to the input order")

# --- 8. retry policy ---------------------------------------------------------

mod.MAX_ATTEMPTS = 3  # keep the backoff sleeps short for the test
llm = fresh()
fake.script = [(429, {"error": "rate limited"}),
               (503, {"error": "upstream"}),
               (200, fake._completion("<signal>NTST</signal>"))]
check(llm.inference("x") == "<signal>NTST</signal>",
      "a 429 then a 503 are retried, and the third attempt succeeds")

llm = fresh()
fake.script = [(401, {"error": "bad key"})]
try:
    llm.inference("x")
    check(False, "a 401 fails fast instead of being retried")
except RuntimeError as exc:
    check(len(fake.requests) == 1 and "401" in str(exc),
          "a 401 fails fast instead of being retried")

llm = fresh()
fake.script = [(429, {"error": "rate limited"})]
try:
    llm.inference("x")
    check(False, "persistent rate limiting eventually raises")
except RuntimeError:
    check(len(fake.requests) == mod.MAX_ATTEMPTS,
          f"persistent rate limiting raises after {mod.MAX_ATTEMPTS} attempts")

# An HTTP 200 whose body carries an error is still a failure.
llm = fresh()
fake.script = [(200, {"error": {"message": "upstream exploded"}})]
try:
    llm.inference("x")
    check(False, "an error inside a 200 body is treated as a failure")
except RuntimeError:
    check(True, "an error inside a 200 body is treated as a failure")

# --- 9. error taxonomy: infra raises, bad model output does not --------------

llm = fresh()
fake.script = [(200, fake._completion(None))]
check(llm.inference("x") == "",
      "an empty completion returns '' (a parse error) rather than raising")
check(runner.parse_llm_signal("") is None,
      "the runner then classifies that as a fallback_parse_error, not an infra error")

fake.shutdown()
restore_real_key()
print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED")
    for msg in failures:
        print(f"  - {msg}")
    sys.exit(1)
print("ALL CHECKS PASSED")
