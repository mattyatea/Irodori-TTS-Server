from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from irodori_openai_tts import runtime as runtime_module
from irodori_openai_tts.config import Settings
from irodori_openai_tts.runtime import RuntimeLoadTimeoutError, RuntimeManager


@pytest.mark.parametrize(
    ("configured_device", "expected_device"),
    [(None, None), ("cpu", "cpu")],
)
def test_runtime_passes_optional_watermark_device_to_core(
    tmp_path, monkeypatch, configured_device, expected_device
):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"test")
    settings = Settings(
        checkpoint=str(checkpoint),
        model_device="cpu",
        codec_device="cpu",
        watermark_device=configured_device,
        _env_file=None,
    )
    manager = RuntimeManager(settings)
    captured = {}
    loaded_runtime = object()

    def fake_runtime_key(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(runtime_module, "RuntimeKey", fake_runtime_key)
    monkeypatch.setattr(
        runtime_module.InferenceRuntime,
        "from_key",
        staticmethod(lambda _key: loaded_runtime),
    )

    assert manager.get() is loaded_runtime
    assert captured["watermark_device"] == expected_device


def test_runtime_load_timeout_while_another_thread_is_loading(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"test")
    settings = Settings(
        checkpoint=str(checkpoint),
        model_device="cpu",
        codec_device="cpu",
        model_load_timeout=0.05,
        _env_file=None,
    )
    manager = RuntimeManager(settings)
    started = threading.Event()
    release = threading.Event()
    loaded_runtime = object()
    errors: list[BaseException] = []

    monkeypatch.setattr(
        runtime_module,
        "RuntimeKey",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    def fake_from_key(_key):
        started.set()
        release.wait(timeout=2)
        return loaded_runtime

    monkeypatch.setattr(
        runtime_module.InferenceRuntime,
        "from_key",
        staticmethod(fake_from_key),
    )

    def load_runtime():
        try:
            assert manager.get() is loaded_runtime
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    thread = threading.Thread(target=load_runtime)
    thread.start()
    assert started.wait(timeout=1)

    with pytest.raises(RuntimeLoadTimeoutError):
        manager.get()

    release.set()
    thread.join(timeout=2)
    assert errors == []
    assert manager.is_loaded
    assert not manager.is_loading


def test_runtime_resolves_local_checkpoint_path(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"test")
    manager = RuntimeManager(Settings(checkpoint=str(checkpoint), _env_file=None))

    assert manager._resolve_checkpoint_path() == str(checkpoint)


def test_runtime_rejects_missing_local_checkpoint(tmp_path):
    manager = RuntimeManager(
        Settings(checkpoint=str(tmp_path / "missing.safetensors"), _env_file=None)
    )

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        manager._resolve_checkpoint_path()


def test_runtime_downloads_hf_checkpoint_when_local_checkpoint_is_unset(monkeypatch):
    manager = RuntimeManager(Settings(hf_checkpoint="owner/repo", _env_file=None))

    def fake_download(*, repo_id, filename):
        assert repo_id == "owner/repo"
        assert filename == "model.safetensors"
        return "/cache/model.safetensors"

    monkeypatch.setattr(runtime_module, "hf_hub_download", fake_download)

    assert manager._resolve_checkpoint_path() == "/cache/model.safetensors"
