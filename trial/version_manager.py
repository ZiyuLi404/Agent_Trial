import json
import os

# Resolve relative to this module's directory so the version state file is found
# regardless of the current working directory.
VERSION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "current_version.json")


def load_version():
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE) as f:
            return json.load(f)
    return {"version_id": "v0", "model_name": "", "prompt_version": "", "tool_version": ""}


def save_version(v):
    with open(VERSION_FILE, "w") as f:
        json.dump(v, f, indent=2)


def open_new_version(version_id, model_name, prompt_version, tool_version):
    """Open a new trial version epoch. Call this manually when the agent changes."""
    v = {
        "version_id": version_id,
        "model_name": model_name,
        "prompt_version": prompt_version,
        "tool_version": tool_version,
    }
    save_version(v)
    return v


def get_current_version():
    return load_version()
