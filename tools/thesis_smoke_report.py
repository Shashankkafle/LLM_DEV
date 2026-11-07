"""Print what a thesis-arm smoke run sent and got back.

    python tools/thesis_smoke_report.py <run_dir>

  (a) one full prompt: the chat messages the client sent, and the exact text the
      server rendered from them with its chat template (via /tokenize +
      /detokenize), checked against paper 2's chatml layout;
  (b) one parsed response: the raw completion and the phase the runner took;
  (c) the first lines of the per-step log.

Fails (exit 1) if any decision in the run carried an incident-report section.
"""
import argparse
import glob
import json
import os
import sys
import urllib.request
from pathlib import Path

REPORT_HEADERS = ("LANE BLOCKAGE REPORT", "DOWNSTREAM BLOCKAGE REPORT")
STEP_LOG_LINES = 8


def load_decisions(run_dir):
    decisions = []
    for path in sorted(glob.glob(os.path.join(run_dir, "intersection_*", "decisions.jsonl"))):
        with open(path, encoding="utf-8") as handle:
            decisions += [json.loads(line) for line in handle if line.strip()]
    return decisions


def queried(decisions):
    """Decisions that actually went to the model, earliest first."""
    sent = [d for d in decisions if (d.get("llm_input") or {}).get("user_prompt")]
    return sorted(sent, key=lambda d: (d["step"], d["intersection_id"]))


def report_leaks(decisions):
    """Decisions whose prompt carried an incident report, by either signal."""
    leaks = []
    for d in decisions:
        prompt = d["llm_input"]["user_prompt"]
        if d.get("blockage_info_in_prompt") or any(h in prompt for h in REPORT_HEADERS):
            leaks.append((d["intersection_id"], d["step"]))
    return leaks


def post_json(url, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def render_on_server(base_url, model, messages):
    """The prompt text the server builds from these messages, or None if the
    server has no /tokenize + /detokenize."""
    root = base_url[: -len("/v1")] if base_url.endswith("/v1") else base_url
    try:
        tokens = post_json(f"{root}/tokenize", {
            "model": model, "messages": messages, "add_generation_prompt": True,
        })["tokens"]
        return post_json(f"{root}/detokenize", {"model": model, "tokens": tokens})["prompt"]
    except Exception as exc:
        print(f"[warning] could not render on the server: {exc}")
        return None


def paper2_chatml(system_prompt, user_prompt):
    """The layout paper 2's in-process chat template produced (its manifest's
    example_formatted_prompt)."""
    return (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def show_prompt(manifest, decision):
    llm = manifest["llm"]
    messages = [{"role": "system", "content": llm["system_prompt"]},
                {"role": "user", "content": decision["llm_input"]["user_prompt"]}]
    section(f"(a) Prompt sent for {decision['intersection_id']} at step {decision['step']}")
    print("--- chat messages (client side) ---")
    print(json.dumps(messages, indent=2, ensure_ascii=False))
    rendered = render_on_server(llm["base_url"], llm["model"], messages)
    if rendered is not None:
        print("\n--- as rendered by the server's chat template ---")
        print(rendered)
        expected = paper2_chatml(messages[0]["content"], messages[1]["content"])
        print(f"\nmatches paper 2 chatml layout byte for byte: {rendered == expected}")


def show_response(decision):
    section("(b) Parsed response for the same decision")
    print("--- raw completion ---")
    print((decision.get("llm_output") or {}).get("raw_text"))
    print("\n--- what the runner did with it ---")
    print(json.dumps(decision.get("phase_action"), indent=2))
    print(json.dumps(decision.get("metrics"), indent=2))


def show_step_log(run_dir):
    section(f"(c) First {STEP_LOG_LINES} lines of step_summaries.jsonl")
    with open(Path(run_dir) / "step_summaries.jsonl", encoding="utf-8") as handle:
        for _, line in zip(range(STEP_LOG_LINES), handle):
            print(line.rstrip())


def show_totals(decisions, sent, leaks):
    section("Totals")
    fallbacks = sum(1 for d in sent if (d.get("phase_action") or {}).get("fallback_applied"))
    print(f"decision records: {len(decisions)}   sent to the model: {len(sent)}   "
          f"fallbacks: {fallbacks}")
    print(f"decisions with an incident-report section: {len(leaks)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir")
    args = parser.parse_args()

    manifest = json.loads((Path(args.run_dir) / "run_manifest.json").read_text(encoding="utf-8"))
    decisions = load_decisions(args.run_dir)
    sent = queried(decisions)
    if not sent:
        sys.exit("no decision in this run was sent to the model")
    leaks = report_leaks(sent)

    show_prompt(manifest, sent[0])
    show_response(sent[0])
    show_step_log(args.run_dir)
    show_totals(decisions, sent, leaks)
    if leaks:
        sys.exit(f"FAIL: incident report present in {len(leaks)} prompts, first {leaks[:5]}")
    print("PASS: no incident-report section in any prompt")


if __name__ == "__main__":
    main()
