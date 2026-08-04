import re
import tomllib
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).parent.parent
PRECOMMIT_FILE_PATH = REPOSITORY_ROOT / ".pre-commit-config.yaml"
PYPROJECT_FILE_PATH = REPOSITORY_ROOT / "pyproject.toml"
WORKFLOW_DIRECTORY = REPOSITORY_ROOT / ".github" / "workflows"


def _exact_dev_pin(package: str) -> str:
    """Return an exact development dependency pin from the project metadata."""
    with PYPROJECT_FILE_PATH.open("rb") as file_handle:
        dependencies = tomllib.load(file_handle)["dependency-groups"]["dev"]

    prefix = f"{package}=="
    matches = [dependency.removeprefix(prefix) for dependency in dependencies if dependency.startswith(prefix)]
    assert len(matches) == 1, f"Expected one exact {package} pin, found {matches}"
    return matches[0]


def _hook_repositories() -> dict[str, dict[str, object]]:
    """Return pre-commit repository definitions keyed by repository URL."""
    with PRECOMMIT_FILE_PATH.open(encoding="utf-8") as file_handle:
        configuration = yaml.safe_load(file_handle)
    return {repository["repo"]: repository for repository in configuration["repos"]}


def test_python_quality_hook_versions_match_project_pins() -> None:
    """Ruff and Pyrefly hooks use the exact versions installed for project development."""
    repositories = _hook_repositories()
    ruff = repositories["https://github.com/astral-sh/ruff-pre-commit"]
    pyrefly = repositories["https://github.com/facebook/pyrefly-pre-commit"]

    assert ruff["rev"] == f"v{_exact_dev_pin('ruff')}"
    assert pyrefly["rev"] == _exact_dev_pin("pyrefly")
    assert {hook["id"] for hook in ruff["hooks"]} == {"ruff-check", "ruff-format"}
    assert pyrefly["hooks"] == [
        {
            "id": "pyrefly-check",
            "name": "Pyrefly (type checking)",
            "entry": "uv run --no-sync pyrefly check",
            "language": "system",
        },
    ]


def test_ci_quality_action_versions_are_uniform() -> None:
    """Every workflow uses one reviewed release for each shared quality action."""
    expected_versions = {
        "actions/checkout": "v7",
        "astral-sh/setup-uv": "v9",
        "hadolint/hadolint-action": "v3.4.0",
    }
    discovered_versions = {action: set() for action in expected_versions}

    for workflow_path in WORKFLOW_DIRECTORY.glob("*.yml"):
        workflow = workflow_path.read_text(encoding="utf-8")
        for action in expected_versions:
            discovered_versions[action].update(re.findall(rf"uses:\s*{re.escape(action)}@([^\s]+)", workflow))

    assert discovered_versions == {action: {version} for action, version in expected_versions.items()}
