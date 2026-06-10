from pathlib import Path

import pytest
import yaml

PRECOMMIT_FILE_PATH = Path(__file__).parent.parent / ".pre-commit-config.yaml"


@pytest.mark.skip(reason="This test is meant to be run manually to check for version mismatches.")
def test_pre_commit_dep_versions(
    horde_dependency_versions: dict[str, str],
    tracked_dependencies: list[str],
) -> None:
    """Check that the versions of horde deps. in .pre-commit-config.yaml match the versions in pyproject.toml.

    See the `tracked_dependencies` fixture for the dependencies tracked.

    Args:
        horde_dependency_versions (list[tuple[str, str]]): The versions of the dependencies in pyproject.toml.
        tracked_dependencies (list[str]): The dependencies to track.

    """
    # Load the pre-commit config
    with open(PRECOMMIT_FILE_PATH) as f:
        precommit_config = yaml.safe_load(f)

    # Initialize a dictionary to hold the versions of the dependencies
    versions = dict.fromkeys(tracked_dependencies)

    # Extract versions from the pre-commit config
    for repo in precommit_config["repos"]:
        if "mypy" in repo["repo"]:
            for dep in repo["hooks"][0]["additional_dependencies"]:
                # Find which tracked dependency this dep string refers to
                matched_dep = next((name for name in versions if name in dep), None)
                if matched_dep is None:
                    continue

                try:
                    if "==" in dep:
                        dep_version = dep.split("==")[1]
                    elif "~=" in dep:
                        dep_version = dep.split("~=")[1]
                    elif ">=" in dep:
                        dep_version = dep.split(">=")[1]
                    else:
                        raise ValueError(f"Unsupported version pin: {dep}")
                except Exception as e:
                    raise ValueError(
                        f"Failed to split dependency: {dep}. Are you missing an exact version pin?",
                    ) from e
                versions[matched_dep] = dep_version

    # Ensure all versions were found
    assert all(
        version is not None for version in versions.values()
    ), f"Some dependencies are missing their versions.\n{versions}"

    # Check if the versions match
    matches = sum(1 for dep, version in horde_dependency_versions.items() if versions.get(dep) == version)

    assert matches == len(
        horde_dependency_versions,
    ), f"Not all dependency versions match.\n`.pre-commit-config.yaml: {versions}"
