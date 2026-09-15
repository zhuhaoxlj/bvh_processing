from io import BytesIO
from pathlib import Path

import pytest

from bvh_processing.config import Settings
from bvh_processing.services import mdm_transition
from bvh_processing.services.download import DownloadedBvh
from bvh_processing.vendor.mdm.merge import GUIDANCE_PARAM, TEXT_CONDITION


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
    assert [(clip.filename, clip.content) for clip in clips] == [
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
