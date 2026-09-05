"""A stand-in vLLM server for offline tests: /v1/models, /version, /tokenize,
/metrics and /v1/chat/completions, on a random local port.

Completions come from a script of (status, body); the last entry repeats once
the script is exhausted. delay_s slows every completion so tests can observe
timing and concurrency. /metrics is a small synthetic Prometheus page driven
by the live counters, with families toggled by `metrics_kv`.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SERVED = "qwen2.5_14b"


class FakeVLLM:
    def __init__(self):
        self.script = [(200, self._completion("<signal>ETWT</signal>"))]
        self.requests = []
        self.served = [SERVED]
        self.has_version = True
        self.has_tokenize = True
        # Token counts /tokenize reports for enable_thinking true/false. Equal
        # counts are how a template that ignores the variable presents itself.
        self.tokenize_counts = {True: 20, False: 12}
        self.delay_s = 0.0
        self.metrics_kv = True
        self.inflight = 0
        self.completed = 0
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

    def metrics_text(self):
        with self._lock:
            running, done = self.inflight, self.completed
        lines = [
            "# HELP vllm:num_requests_running Number of requests in model execution batches.",
            "# TYPE vllm:num_requests_running gauge",
            f'vllm:num_requests_running{{model_name="{SERVED}"}} {running}.0',
            f'vllm:num_requests_waiting{{model_name="{SERVED}"}} 0.0',
            f'vllm:num_preemptions_total{{model_name="{SERVED}"}} 0.0',
            f'vllm:prompt_tokens_total{{model_name="{SERVED}"}} {done * 759}.0',
            f'vllm:generation_tokens_total{{model_name="{SERVED}"}} {done * 42}.0',
            f'vllm:request_success_total{{finished_reason="stop",model_name="{SERVED}"}} {done}.0',
            f'vllm:time_to_first_token_seconds_bucket{{le="0.1",model_name="{SERVED}"}} {done}.0',
            f'vllm:time_to_first_token_seconds_bucket{{le="1.0",model_name="{SERVED}"}} {done}.0',
            f'vllm:time_to_first_token_seconds_bucket{{le="+Inf",model_name="{SERVED}"}} {done}.0',
            f'vllm:time_to_first_token_seconds_count{{model_name="{SERVED}"}} {done}.0',
            "this line is deliberately malformed",
            f'vllm:something_nan{{model_name="{SERVED}"}} NaN',
        ]
        if self.metrics_kv:
            lines.append(f'vllm:kv_cache_usage_perc{{model_name="{SERVED}"}} 0.061')
        return "\n".join(lines) + "\n"

    def _handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, status, body, content_type="application/json"):
                encoded = (body if isinstance(body, str) else json.dumps(body)).encode()
                self.send_response(status)
                self.send_header("Content-Type", content_type)
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
                elif self.path == "/metrics":
                    self._reply(200, outer.metrics_text(), "text/plain")
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
                with outer._lock:
                    outer.inflight += 1
                try:
                    time.sleep(outer.delay_s)
                    self._reply(*outer._next(payload))
                finally:
                    with outer._lock:
                        outer.inflight -= 1
                        outer.completed += 1

            def log_message(self, *args):
                pass

        return Handler

    def completions(self):
        return [r for r in self.requests if r.get("_path") == "/v1/chat/completions"]

    def reset(self, script=None):
        self.script = script or [(200, self._completion("<signal>ETWT</signal>"))]
        self.requests.clear()

    def shutdown(self):
        self._server.shutdown()
