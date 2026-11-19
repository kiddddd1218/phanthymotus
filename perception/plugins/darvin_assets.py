"""Cache pinned Darvin assets at image build time without loading an engine."""
import argparse
import json
import os
from pathlib import Path


def candidate_spec(name):
    path = Path(__file__).resolve().parents[1] / "data/darvin_candidates.json"
    specs = json.loads(path.read_text(encoding="utf-8"))
    if name not in specs:
        raise ValueError("unknown Darvin candidate: " + str(name))
    return specs[name]


def ensure_candidate_bundle(name, spec):
    from utils.model_downloader import ensure_verified_bundle

    directory = Path(os.environ.get("ACTION_MODEL_DIR", "/models/action")) / spec["cache_directory"]
    return ensure_verified_bundle("darvin314-" + name, str(directory), spec["base_url"], spec["files"])


def prefetch(config_path):
    import yaml

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    pose = config.get("plugins", {}).get("pose", {})
    name = pose.get("darvin_candidate", "")
    if not name:
        return {}  # Preserve the original image behavior without a selected candidate.
    spec = candidate_spec(name)
    if pose.get("action_backend", "hybrid") != spec["backend"]:
        raise ValueError("Darvin candidate and action_backend disagree")
    if pose.get("official_action_manifest"):
        raise ValueError("choose either a pinned Darvin candidate or a local official manifest")
    paths = ensure_candidate_bundle(name, spec)
    print("[build] verified Darvin candidate %s: %d files" % (name, len(paths)))
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="perception YAML configuration")
    prefetch(parser.parse_args().config)
