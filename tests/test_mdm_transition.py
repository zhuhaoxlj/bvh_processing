import subprocess
from io import BytesIO
from pathlib import Path

import pytest

from bvh_processing.config import Settings
from bvh_processing.services.download import DownloadedBvh
from bvh_processing.services.mdm_transition import generate_mdm_merge


def _downloaded(name: str, content: bytes) -> DownloadedBvh:
    return DownloadedBvh(BytesIO(content), name, len(content))


def test_mdm_merge_runs_configured_virtualenv_and_returns_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "mdm"
    python = project / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    calls: list[tuple[list[str], Path]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append((command, Path(str(kwargs["cwd"]))))
        output = Path(command[command.index("--output") + 1])
        output.write_bytes(b"generated-bvh")
        return subprocess.CompletedProcess(command, 0, "{}", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    settings = Settings(
        mdm_project_root=str(project),
        mdm_python=str(python),
        mdm_seed=42,
        mdm_source_scale=0.001,
        mdm_source_up_axis="Z",
    )

    result = generate_mdm_merge(
        [_downloaded("a.bvh", b"A"), _downloaded("b.bvh", b"B")],
        [0.75],
        settings,
    )

    assert result == b"generated-bvh"
    command, cwd = calls[0]
    assert command[0] == str(python.absolute())
    assert command[1:3] == ["-m", "bvh_workbench.merge_cli"]
    assert command[command.index("--transition-seconds") + 1] == "0.75"
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--scale") + 1] == "0.001"
    assert command[command.index("--up-axis") + 1] == "Z"
    assert cwd == project.resolve()


def test_mdm_merge_reports_subprocess_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "mdm"
    python = project / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, "", "unsupported skeleton"
        ),
    )

    with pytest.raises(ValueError, match="unsupported skeleton"):
        generate_mdm_merge(
            [_downloaded("a.bvh", b"A"), _downloaded("b.bvh", b"B")],
            [1.0],
            Settings(mdm_project_root=str(project), mdm_python=str(python)),
        )
