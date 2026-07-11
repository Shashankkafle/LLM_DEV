import json


def get_phase_name(config, phase_string):
    if phase_string == config["global_settings"]["all_red_state"]:
        return "ALL_RED"

    for phase_name, phase_data in config["phases"].items():
        if phase_string in (phase_data["green"], phase_data["yellow"]):
            return phase_name

    print(f"Warning: Phase string '{phase_string}' does not match any known phase in the configuration.")



def append_jsonl(file_path, entry):
    """Append one JSON object as a line to a JSONL file (created on first write)."""
    with open(file_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

