import os
import json


def get_phase_name(config, phase_string):
    if phase_string == config["global_settings"]["all_red_state"]:
        return "ALL_RED"

    for phase_name, phase_data in config["phases"].items():
        if phase_string in (phase_data["green"], phase_data["yellow"]):
            return phase_name

    print(f"Warning: Phase string '{phase_string}' does not match any known phase in the configuration.")



def append_to_json_file(file_path, entry):
    """Read an existing JSON array from file_path (or start fresh), append entry, and write back."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(entry)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

