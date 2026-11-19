"""Image packaging and cold/warm cache checks, without network or inference."""
import copy
import hashlib
import io
from pathlib import Path
import shlex
import shutil

import pytest
import yaml

from plugins import darvin_assets, official_action
from utils import model_downloader


@pytest.mark.parametrize("name", ["native60_streaming", "official126_streaming"])
def test_docker_copy_layout_contains_registry_and_taxonomy(tmp_path, monkeypatch, name):
    root = Path(__file__).resolve().parents[2]
    recipe = (root / "perception/Dockerfile.jetson").read_text()
    # Reconstruct the image's application COPY destinations, not a source checkout.
    for line in recipe.splitlines():
        if not line.startswith("COPY "):
            continue
        fields = shlex.split(line)
        if len(fields) != 3 or not fields[2].startswith("/work/"):
            continue
        source = root / fields[1]
        destination = tmp_path / fields[2].lstrip("/")
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    monkeypatch.setattr(darvin_assets, "__file__", str(tmp_path / "work/plugins/darvin_assets.py"))
    monkeypatch.setattr(official_action, "__file__", str(tmp_path / "work/plugins/official_action.py"))
    assert darvin_assets.candidate_spec(name)["files"]
    assert len(official_action.canonical_labels()) == 126
    assert "ENV ACTION_MODEL_DIR=/work/models/action" in recipe
    assert "RUN python3 -m plugins.darvin_assets /work/config.yaml" in recipe


def synthetic_bundle(tmp_path, monkeypatch):
    body = b"fixture model contents"
    spec = dict(backend="hybrid", cache_directory="test-revision", base_url="https://invalid.example",
                files={"fixture.bin": dict(size=len(body), sha256=hashlib.sha256(body).hexdigest())})
    monkeypatch.setattr(darvin_assets, "candidate_spec", lambda name: copy.deepcopy(spec))
    monkeypatch.setenv("ACTION_MODEL_DIR", str(tmp_path / "models"))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"plugins": {"pose": {
        "darvin_candidate": "fixture", "action_backend": "hybrid"}}}))
    return config, body


def test_prefetch_downloads_verifies_then_reuses_offline(tmp_path, monkeypatch):
    config, body = synthetic_bundle(tmp_path, monkeypatch)
    calls = []

    def response(url, **kwargs):
        calls.append(url)
        return io.BytesIO(body)

    monkeypatch.setattr(model_downloader, "urlopen", response)
    paths = darvin_assets.prefetch(config)
    assert Path(paths["fixture.bin"]).read_bytes() == body
    assert calls

    def no_network(*args, **kwargs):
        pytest.fail("a complete verified build cache must work offline")

    monkeypatch.setattr(model_downloader, "urlopen", no_network)
    assert darvin_assets.prefetch(config) == paths
    spec = darvin_assets.candidate_spec("fixture")
    assert darvin_assets.ensure_candidate_bundle("fixture", spec) == paths


@pytest.mark.parametrize("body", [b"", b"X" * len(b"fixture model contents")])
def test_empty_or_wrong_hash_download_fails_prefetch(tmp_path, monkeypatch, body):
    config, _ = synthetic_bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(model_downloader, "urlopen", lambda *a, **kw: io.BytesIO(body))
    monkeypatch.setattr(model_downloader.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="failed to download"):
        darvin_assets.prefetch(config)
    assert not (tmp_path / "models/test-revision/fixture.bin").exists()


def test_mismatched_config_fails_before_download(tmp_path, monkeypatch):
    config, _ = synthetic_bundle(tmp_path, monkeypatch)
    config.write_text(yaml.safe_dump({"plugins": {"pose": {
        "darvin_candidate": "fixture", "action_backend": "official126"}}}))
    monkeypatch.setattr(darvin_assets, "ensure_candidate_bundle", lambda *a: pytest.fail("download attempted"))
    with pytest.raises(ValueError, match="disagree"):
        darvin_assets.prefetch(config)


def test_unselected_candidate_preserves_original_build(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("plugins: {pose: {action_backend: hybrid}}\n")
    monkeypatch.setattr(darvin_assets, "candidate_spec", lambda *a: pytest.fail("candidate selected"))
    assert darvin_assets.prefetch(config) == {}
