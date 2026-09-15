from io import BytesIO
from pathlib import Path

import pytest

from bvh_processing.config import Settings
from bvh_processing.services import mdm_transition
from bvh_processing.services.download import DownloadedBvh
from bvh_processing.vendor.mdm.merge import GUIDANCE_PARAM, TEXT_CONDITION
from bvh_processing.vendor.mdm.workbench.transitions import MDMEngine


def _downloaded(name: str, content: bytes) -> DownloadedBvh:
    return DownloadedBvh(BytesIO(content), name, len(content))


def test_mdm_merge_calls_bundled_module_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resource_root = tmp_path / "mdm-assets"
    engine = object()
    calls: list[tuple[object, object]] = []

    def fake_engine_for(path: Path) -> object:
        calls.append(("resource_root", path))
        return engine

    def fake_merge(clips, intervals, **kwargs) -> bytes:
        calls.append((clips, (intervals, kwargs)))
        return b"generated-bvh"

    monkeypatch.setattr(mdm_transition, "_engine_for", fake_engine_for)
    monkeypatch.setattr(mdm_transition, "merge_bvh_clips", fake_merge)
    settings = Settings(
        mdm_resource_root=str(resource_root),
        mdm_seed=42,
        mdm_source_scale=0.001,
        mdm_source_up_axis="Z",
    )

    result = mdm_transition.generate_mdm_merge(
        [_downloaded("a.bvh", b"A"), _downloaded("b.bvh", b"B")],
        [0.75],
        settings,
    )

    assert result == b"generated-bvh"
    assert calls[0] == ("resource_root", resource_root)
    clips, (intervals, kwargs) = calls[1]
    assert [(clip.filename, clip.read()) for clip in clips] == [
        ("a.bvh", b"A"),
        ("b.bvh", b"B"),
    ]
    assert intervals == [0.75]
    assert kwargs == {
        "engine": engine,
        "seed": 42,
        "scale": 0.001,
        "up_axis": "Z",
    }


def test_mdm_merge_rejects_empty_module_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mdm_transition, "_engine_for", lambda _path: object())
    monkeypatch.setattr(mdm_transition, "merge_bvh_clips", lambda *args, **kwargs: b"")

    with pytest.raises(ValueError, match="空 BVH"):
        mdm_transition.generate_mdm_merge(
            [_downloaded("a.bvh", b"A"), _downloaded("b.bvh", b"B")],
            [1.0],
            Settings(mdm_resource_root=str(tmp_path)),
        )


def test_mdm_conditioning_is_fixed() -> None:
    assert TEXT_CONDITION == "A short, direct interpolation between the two poses."
    assert GUIDANCE_PARAM == 2.5


def test_mdm_engine_rejects_unresolved_lfs_pointers(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint/model000600000.pt"
    humanml = tmp_path / "humanml"
    text_encoder = tmp_path / "text_encoder"
    checkpoint.parent.mkdir()
    (humanml / "new_joints").mkdir(parents=True)
    text_encoder.mkdir()
    pointer = b"version https://git-lfs.github.com/spec/v1\n"
    checkpoint.write_bytes(pointer)
    checkpoint.with_name("args.json").write_text("{}")
    (humanml / "Mean.npy").write_bytes(b"x")
    (humanml / "Std.npy").write_bytes(b"x")
    (humanml / "new_joints/000021.npy").write_bytes(b"x")
    (text_encoder / "config.json").write_text("{}")
    (text_encoder / "model.safetensors").write_bytes(pointer)
    (text_encoder / "tokenizer.json").write_text("{}")

    engine = MDMEngine(checkpoint, humanml, text_encoder)

    assert engine.status()["available"] is False
    assert engine.status()["invalid"] == [
        str(checkpoint),
        str(text_encoder / "model.safetensors"),
    ]
    with pytest.raises(ValueError, match="git lfs pull"):
        engine._load()
