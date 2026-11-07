"""Add the thesis LLM arm's identity block to a finished run's manifest.

    python tools/thesis_arm_manifest.py <run_dir> --scenario cross_r1200 --seed 1 --horizon 7200

Writes a "thesis_arm" block into <run_dir>/run_manifest.json. The runner is left
untouched (the row is defined as the paper 2 setup), so this runs after it exits,
crashed or not.

The prompt hash covers what the model is shown: the system prompt the run
actually sent (as recorded in the manifest) plus the source of
utils/prompt_builder.py, which renders the user prompt. Line endings are
normalized so a Windows checkout hashes the same as the Linux box.
"""
import argparse
import hashlib
import json
from pathlib import Path

ARM = "llm_zeroshot_noreport_paper2prompt"
PROMPT_BUILDER = Path("utils/prompt_builder.py")

# The paper 2 runs (logs/llm_real_*/..._202607*) all ran on this commit,
# in-process HF transformers fp16, one run at a time.
PAPER2_COMMIT = "15473b8ae3b228358143e52179b1bd518fbf04bc"
# prompt_hash() of LLM_SYSTEM_PROMPT and utils/prompt_builder.py at PAPER2_COMMIT.
PAPER2_PROMPT_HASH = "553b90b80772707052f4cb84468f9fa48dfe60f1d9f6997525e83575c5359edd"

NOTES = [
    "Inference backend differs from paper 2: vLLM server over HTTP (chat template "
    "applied server-side) instead of in-process HF transformers. Same weights, "
    "system prompt, prompt builder, parser, max_new_tokens and greedy decoding.",
    "Weights are the fine-tuned merged Qwen2.5-14B, as in paper 2 -- not the stock "
    "Instruct model.",
    "Paper 2 manifests record git_dirty=true; their uncommitted edits were not saved.",
    "The control loop stops early once SUMO expects no more vehicles, so "
    "last_sim_step can be below horizon_s.",
]


def normalized_bytes(text):
    return text.replace("\r\n", "\n").encode("utf-8")


def prompt_hash(system_prompt, prompt_builder_source):
    digest = hashlib.sha256()
    digest.update(normalized_bytes(system_prompt))
    digest.update(b"\n\0\n")
    digest.update(normalized_bytes(prompt_builder_source))
    return digest.hexdigest()


def last_sim_step(run_dir):
    """The step of the last line of step_summaries.jsonl, or None if empty."""
    path = Path(run_dir) / "step_summaries.jsonl"
    if not path.exists():
        return None
    last_line = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last_line = line
    if last_line is None:
        return None
    return json.loads(last_line)["step"]


def build_block(manifest, run_dir, scenario, seed, horizon):
    llm = manifest.get("llm") or {}
    server = llm.get("server") or {}
    model_entry = server.get("model_entry") or {}
    provenance = manifest.get("provenance") or {}
    system_prompt = llm.get("system_prompt") or ""
    this_hash = prompt_hash(system_prompt, PROMPT_BUILDER.read_text(encoding="utf-8"))
    step = last_sim_step(run_dir)
    return {
        "arm": ARM,
        "scenario": scenario,
        "seed": seed,
        "horizon_s": horizon,
        "last_sim_step": step,
        "stopped_early": step is not None and step + 1 < horizon,
        "model": llm.get("model"),
        "weights": model_entry.get("root"),
        "server_url": llm.get("base_url"),
        "server_version": server.get("version"),
        "serve_cmd": llm.get("serve_cmd"),
        "prompt_hash": this_hash,
        "prompt_hash_matches_paper2": this_hash == PAPER2_PROMPT_HASH,
        "git_commit": provenance.get("git_commit"),
        "git_dirty": provenance.get("git_dirty"),
        "started_at": manifest.get("started_at"),
        "ended_at": manifest.get("ended_at"),
        "wall_clock_duration_s": manifest.get("wall_clock_duration_s"),
        "status": manifest.get("status"),
        "paper2_reference": {"git_commit": PAPER2_COMMIT,
                             "backend": "hf in-process, torch.float16"},
        "notes": NOTES,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    args = parser.parse_args()

    manifest_path = Path(args.run_dir) / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    block = build_block(manifest, args.run_dir, args.scenario, args.seed, args.horizon)
    manifest["thesis_arm"] = block
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[thesis_arm] {manifest_path}: status={block['status']} "
          f"last_sim_step={block['last_sim_step']} "
          f"prompt_hash_matches_paper2={block['prompt_hash_matches_paper2']}")


if __name__ == "__main__":
    main()
