"""Offline checks for the vLLM backend (models_inference/LLM/http_llm).

Runs the real client against a local stand-in server via VLLM_BASE_URL, so
preflight, the reasoning split, the thinking-variable probe and the scheme
dispatch are all exercised without a GPU, weights, or a server.

The load-bearing parts are the two silent-failure guards: a chain of thought
that survives the split reaches parse_llm_signal and picks a phase the model
rejected, and a --reasoning flag the chat template ignores would leave a model
thinking in an arm that is supposed to have thinking off. Both must be loud.

The OpenRouter half of the same module is covered by
tests/smoke_openrouter_backend.py.

Run: python tests/smoke_http_backends.py
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


SERVED = "qwen2.5_14b"


class FakeVLLM:
    """A stand-in vLLM server: /v1/models, /version, /tokenize and
    /v1/chat/completions. Completions come from a script of (status, body); the
    last entry repeats once the script is exhausted."""

    def __init__(self):
        self.script = [(200, self._completion("<signal>ETWT</signal>"))]
        self.requests = []
        self.served = [SERVED]
        self.has_version = True
        self.has_tokenize = True
        # Token counts /tokenize reports for enable_thinking true/false. Equal
        # counts are how a template that ignores the variable presents itself.
        self.tokenize_counts = {True: 20, False: 12}
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}/v1"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @staticmethod
    def _completion(content, reasoning_content=None, prompt_tokens=759,
                    completion_tokens=42, finish_reason="stop"):
        message = {"role": "assistant", "content": content}
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        return {
            "model": SERVED,
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens},
        }

    def _next(self, payload):
        with self._lock:
            self.requests.append(payload)
            return self.script[0] if len(self.script) == 1 else self.script.pop(0)

    def _handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, status, body):
                encoded = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def do_GET(self):
                if self.path == "/v1/models":
                    self._reply(200, {"data": [
                        {"id": name, "root": f"/models/{name}",
                         "max_model_len": 32768} for name in outer.served]})
                elif self.path == "/version" and outer.has_version:
                    self._reply(200, {"version": "0.11.0"})
                else:
                    self._reply(404, {"detail": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                payload["_path"] = self.path
                payload["_auth"] = self.headers.get("Authorization")
                if self.path == "/tokenize":
                    with outer._lock:
                        outer.requests.append(payload)
                    if not outer.has_tokenize:
                        return self._reply(404, {"detail": "not found"})
                    thinking = (payload.get("chat_template_kwargs") or {}).get(
                        "enable_thinking")
                    return self._reply(200, {"count": outer.tokenize_counts[thinking]})
                self._reply(*outer._next(payload))

            def log_message(self, *args):
                pass

        return Handler

    def completions(self):
        return [r for r in self.requests if r.get("_path") == "/v1/chat/completions"]

    def reset(self, script=None):
        self.script = script or [(200, self._completion("<signal>ETWT</signal>"))]
        self.requests.clear()


fake = FakeVLLM()
os.environ["VLLM_BASE_URL"] = fake.base_url
os.environ.pop("VLLM_API_KEY", None)

from models_inference.LLM.http_llm import VLLMLLM, OpenRouterLLM  # noqa: E402
import runner  # noqa: E402

MODEL = f"vllm:{SERVED}"


def fresh(script=None, **kwargs):
    """A ready-to-use client, with the fake reset and the preflight dropped."""
    fake.reset(script)
    llm = VLLMLLM(MODEL, **kwargs)
    llm.initialize_llm()
    fake.requests.clear()
    return llm


# --- 1. scheme dispatch ------------------------------------------------------

check(isinstance(runner.build_llm(MODEL), VLLMLLM),
      "build_llm routes a vllm: path to the vLLM backend")
check(isinstance(runner.build_llm("openrouter:google/gemma-3-27b-it"), OpenRouterLLM),
      "build_llm still routes an openrouter: path to the hosted backend")
try:
    runner.build_llm("~/models/merged/qwen2.5_14b")
    check(False, "an unprefixed path is rejected with the retirement message")
except ValueError as exc:
    check("vllm serve" in str(exc) and "retired" in str(exc),
          "an unprefixed path is rejected with the retirement message")

# --- 2. no API key is required -----------------------------------------------

llm = fresh()
llm.inference("x")
check(llm.api_key is None, "a vLLM server needs no API key")
check(fake.completions() and all(r["_auth"] is None for r in fake.completions()),
      "no Authorization header is sent when no key is set")

# --- 3. preflight proves the model is served, and records the server ---------

llm = VLLMLLM(MODEL)
llm.initialize_llm()
check(llm.server_info.get("version") == "0.11.0",
      "preflight records the server version from /version")
check(llm.server_info["model_entry"]["root"] == f"/models/{SERVED}",
      "preflight records what the server actually loaded (model_entry.root)")
check(llm.describe()["server"]["model_entry"]["max_model_len"] == 32768,
      "describe() carries the server block into the manifest")
check(llm.describe()["backend"] == "vllm", "describe() names the backend")

fake.served = ["some-other-model"]
try:
    VLLMLLM(MODEL).initialize_llm()
    check(False, "an unserved model fails preflight, naming what IS served")
except RuntimeError as exc:
    check("some-other-model" in str(exc) and "--llm_path vllm:" in str(exc),
          "an unserved model fails preflight, naming what IS served")
fake.served = [SERVED]

fake.has_version = False
llm = VLLMLLM(MODEL)
llm.initialize_llm()
check(llm.server_info.get("version") is None,
      "a missing /version is a warning, not a failed run")
fake.has_version = True

# --- 4. --reasoning reaches the model as chat_template_kwargs ----------------

llm = fresh()
llm.inference("x")
check("chat_template_kwargs" not in fake.completions()[0],
      "--reasoning auto sends no chat_template_kwargs at all")

llm = fresh(reasoning="off")
llm.inference("x")
check(fake.completions()[0]["chat_template_kwargs"] == {"enable_thinking": False},
      "--reasoning off sends enable_thinking=False")

llm = fresh(reasoning="on")
llm.inference("x")
check(fake.completions()[0]["chat_template_kwargs"] == {"enable_thinking": True},
      "--reasoning on sends enable_thinking=True")
check(llm.describe()["generation"]["extra_payload_sent"] ==
      {"chat_template_kwargs": {"enable_thinking": True}},
      "describe() records what --reasoning actually sent")

# --- 5. a template that ignores the thinking variable fails at startup -------
# This is the guarantee the retired local backend gave by grepping the chat
# template. Equal token counts both ways prove the template never branches.

fake.tokenize_counts = {True: 12, False: 12}
try:
    VLLMLLM(MODEL, reasoning="off").initialize_llm()
    check(False, "a template that ignores enable_thinking aborts at startup")
except ValueError as exc:
    check("silently ignored" in str(exc),
          "a template that ignores enable_thinking aborts at startup")

fake.reset()
VLLMLLM(MODEL, reasoning="auto").initialize_llm()
check(not [r for r in fake.requests if r.get("_path") == "/tokenize"],
      "--reasoning auto skips the probe entirely (no wasted requests)")

fake.tokenize_counts = {True: 20, False: 12}

fake.has_tokenize = False
try:
    VLLMLLM(MODEL, reasoning="off").initialize_llm()
    check(True, "a server without /tokenize warns rather than failing the run")
except Exception as exc:
    check(False, f"a server without /tokenize must warn, not raise ({exc})")
fake.has_tokenize = True

# --- 6. reasoning is separated from the answer -------------------------------

llm = fresh([(200, fake._completion(
    "<think>ETWT looks busy but is blocked</think>\n<signal>NTST</signal>"))])
answer = llm.inference("x")
check(answer == "<signal>NTST</signal>",
      "an inline <think> block is split out of the answer")
check(llm.last_reasoning == "ETWT looks busy but is blocked",
      "the split-out chain of thought is recorded as reasoning")
check(runner.parse_llm_signal(answer) == "NTST",
      "the signal parser sees only the answer, not the reasoning")

llm = fresh([(200, fake._completion("<signal>NTST</signal>",
                                    reasoning_content="server-parsed thought"))])
answer = llm.inference("x")
check(llm.last_reasoning == "server-parsed thought" and answer == "<signal>NTST</signal>",
      "reasoning_content from --reasoning-parser is preferred over the split")

llm = fresh([(200, fake._completion(
    "<|channel>thought ETWT is fine<channel|><signal>ETWT</signal>"))])
answer = llm.inference("x")
check(llm._leaked_reasoning_calls == 1,
      "an unrecognised reasoning delimiter is reported, not silently parsed")

# --- 7. usage and batching ---------------------------------------------------

llm = fresh()
llm.inference("x")
check(llm.last_usage == {"prompt_tokens": 759, "completion_tokens": 42,
                         "reasoning_tokens": None, "finish_reason": "stop"},
      "last_usage keeps the 4-key shape the recorder expects")

llm = fresh()
outputs = llm.inference_batch(["a", "b", "c"])
check(len(outputs) == 3 and len(llm.last_usage_batch) == 3
      and len(llm.last_reasoning_batch) == 3,
      "inference_batch returns one output, usage and reasoning per prompt")
check(len(fake.completions()) == 3,
      "each prompt in a batch is one independent request")
check(llm.inference_batch([]) == [] and llm.last_usage_batch == [],
      "an empty batch makes no request")

# --- 8. quantization is recorded but never sent ------------------------------

llm = fresh(quantization="awq")
llm.inference("x")
check("quantization" not in fake.completions()[0],
      "--quantization changes nothing about the request")
check(llm.describe()["quantization_declared"] == "awq",
      "--quantization is recorded in the manifest as an operator assertion")

fake._server.shutdown()
print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL CHECKS PASSED")
