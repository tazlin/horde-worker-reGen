from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

PYPROJECT_FILE_PATH = Path(__file__).parent.parent / "pyproject.toml"
UV_LOCK_FILE_PATH = Path(__file__).parent.parent / "uv.lock"

# The worker exposes one thin extra per torch build it locks. torch is left unpinned and floats to the
# latest version the horde-engine ranges allow; each extra only routes torch to its wheel index. Older
# torch lines and ROCm are installed ad-hoc (see pyproject.toml), so they are deliberately NOT extras.
# GPU detection picks the build and update-runtime.* run `uv sync --locked --extra <build>`.
# build extra -> the [[tool.uv.index]] it must be routed through.
BUILD_INDEX = {
    "cu126": "pytorch-cu126",
    "cu130": "pytorch-cu130",
    "cu132": "pytorch-cu132",
    "cpu": "pytorch-cpu",
}


def _load_pyproject() -> dict:
    with open(PYPROJECT_FILE_PATH, "rb") as f:
        return tomllib.load(f)


def test_uv_lock_exists() -> None:
    """Check that uv.lock exists and is not empty."""
    assert UV_LOCK_FILE_PATH.exists(), "uv.lock not found — run 'uv lock' to generate it"
    assert UV_LOCK_FILE_PATH.stat().st_size > 0, "uv.lock is empty"


def test_build_extras_list_torch() -> None:
    """Every build extra exists and lists torch (so [tool.uv.sources] routes it to the build's index)."""
    extras = _load_pyproject()["project"]["optional-dependencies"]
    for build in BUILD_INDEX:
        assert build in extras, f"missing build extra '{build}'"
        assert any(
            d == "torch" or d.startswith(("torch==", "torch>", "torch~", "torch<")) for d in extras[build]
        ), f"'{build}' must list torch so it can be routed to {BUILD_INDEX[build]}"


def test_no_stale_leaf_extras() -> None:
    """The old torch<line>-<build> leaf extras must be gone (matrix collapsed to thin build extras)."""
    extras = _load_pyproject()["project"]["optional-dependencies"]
    stale = [name for name in extras if name.startswith("torch2")]
    assert not stale, f"stale leaf extras still present: {stale}"


def test_build_extras_routed_to_matching_index() -> None:
    """Each build extra routes torch to the wheel index that matches it."""
    data = _load_pyproject()
    routes = {(entry["extra"], entry["index"]) for entry in data["tool"]["uv"]["sources"]["torch"]}
    for build, index in BUILD_INDEX.items():
        assert (build, index) in routes, f"'{build}' not routed to {index}"


def test_conflicts_cover_all_builds() -> None:
    """All build extras must be mutually exclusive in a single conflicts group."""
    data = _load_pyproject()
    groups = [{item["extra"] for item in group} for group in data["tool"]["uv"]["conflicts"]]
    assert set(BUILD_INDEX) in groups, "the build extras are not declared as one conflicts group"


def test_pytorch_indexes_defined() -> None:
    """Every index a build routes to must be declared as a [[tool.uv.index]]."""
    index_names = {idx["name"] for idx in _load_pyproject()["tool"]["uv"].get("index", [])}
    for index in set(BUILD_INDEX.values()):
        assert index in index_names, f"no [[tool.uv.index]] named '{index}'"
