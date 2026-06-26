from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SEMANTIC_VERSION = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


def _project_metadata() -> dict[str, object]:
    payload = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = payload.get("project")
    assert isinstance(project, dict)
    return project


def test_release_metadata_is_complete_and_consistent() -> None:
    project = _project_metadata()
    version = project.get("version")
    description = project.get("description")
    readme = project.get("readme")

    assert isinstance(version, str)
    assert SEMANTIC_VERSION.fullmatch(version)
    assert isinstance(description, str)
    assert description.strip()
    assert "bootstrap" not in description.casefold()
    assert readme == "README.md"
    assert (REPOSITORY_ROOT / str(readme)).is_file()

    changelog = (REPOSITORY_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{version}]" in changelog

    lockfile = (REPOSITORY_ROOT / "uv.lock").read_text(encoding="utf-8")
    package_entry = (
        'name = "smart-beauty-resize-preprocessor"\n'
        f'version = "{version}"\n'
        'source = { editable = "." }'
    )
    assert package_entry in lockfile


def test_release_handoff_documents_are_present() -> None:
    required_documents = (
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "docs/integration-guide.md",
        "docs/release-checklist.md",
        "docs/performance-measurement.md",
    )

    for relative_path in required_documents:
        document = REPOSITORY_ROOT / relative_path
        assert document.is_file()
        assert document.read_text(encoding="utf-8").strip()
