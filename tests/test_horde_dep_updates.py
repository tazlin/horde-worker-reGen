from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

PYPROJECT_FILE_PATH = Path(__file__).parent.parent / "pyproject.toml"
UV_LOCK_FILE_PATH = Path(__file__).parent.parent / "uv.lock"

# The worker exposes one thin extra per torch build it locks. torch and torchvision are left unpinned
# and float to the latest version the horde-engine ranges allow; each extra routes BOTH to its wheel
# index so their CUDA builds can never disagree (a torchvision left on generic PyPI pulls a different
# CUDA build, which is the mismatch crash these tests guard against). torchaudio is deliberately NOT
# routed (no +cu132 wheel; audio unsupported) and must stay out of the build extras. Older torch lines
# and ROCm are installed ad-hoc (see pyproject.toml), so they are deliberately NOT extras. GPU detection
# picks the build and update-runtime.* run `uv sync --locked --extra <build>`.
# build extra -> the [[tool.uv.index]] it must be routed through.
BUILD_INDEX = {
    "cu126": "pytorch-cu126",
    "cu130": "pytorch-cu130",
    "cu132": "pytorch-cu132",
    "cpu": "pytorch-cpu",
}
# build extra -> the wheel index URL its torch/torchvision must resolve from in the lock.
BUILD_INDEX_URL = {build: f"https://download.pytorch.org/whl/{build}" for build in BUILD_INDEX}
# The packages that must stay pinned to one build together. torchaudio is intentionally excluded.
ROUTED_PACKAGES = ("torch", "torchvision")


def _load_pyproject() -> dict:
    with open(PYPROJECT_FILE_PATH, "rb") as f:
        return tomllib.load(f)


def _load_lock() -> dict:
    with open(UV_LOCK_FILE_PATH, "rb") as f:
        return tomllib.load(f)


def _dep_name(spec: str) -> str:
    """Strip version/extra/marker decoration from a dependency spec, leaving the bare package name."""
    name = spec.split(";", 1)[0].strip()
    for sep in ("[", "=", ">", "<", "~", "!", " "):
        name = name.split(sep, 1)[0]
    return name.strip()


def _local_build_tag(version: str) -> str | None:
    """Return a wheel's local build tag (``2.12.0+cu132`` -> ``cu132``), or None when untagged."""
    return version.split("+", 1)[1] if "+" in version else None


def _normalize(name: str) -> str:
    """Normalize a distribution name for comparison (``horde_engine`` and ``horde-engine`` are one)."""
    return name.replace("_", "-").lower()


def _requested_extras(spec: str) -> set[str]:
    """Return the extras requested in a dependency spec (the names inside ``[...]``)."""
    before_marker = spec.split(";", 1)[0]
    if "[" not in before_marker:
        return set()
    inside = before_marker.split("[", 1)[1].split("]", 1)[0]
    return {part.strip() for part in inside.split(",") if part.strip()}


def test_uv_lock_exists() -> None:
    """Check that uv.lock exists and is not empty."""
    assert UV_LOCK_FILE_PATH.exists(), "uv.lock not found — run 'uv lock' to generate it"
    assert UV_LOCK_FILE_PATH.stat().st_size > 0, "uv.lock is empty"


def test_build_extras_list_routed_packages() -> None:
    """Every build extra exists and lists torch and torchvision so their per-build routing applies."""
    extras = _load_pyproject()["project"]["optional-dependencies"]
    for build in BUILD_INDEX:
        assert build in extras, f"missing build extra '{build}'"
        names = {_dep_name(d) for d in extras[build]}
        for package in ROUTED_PACKAGES:
            assert package in names, f"'{build}' must list '{package}' so it routes to {BUILD_INDEX[build]}"


def test_torchaudio_not_in_build_extras() -> None:
    """Reject any build extra that lists torchaudio (it has no +cu132 wheel; audio is unsupported)."""
    extras = _load_pyproject()["project"]["optional-dependencies"]
    for build in BUILD_INDEX:
        names = {_dep_name(d) for d in extras[build]}
        assert "torchaudio" not in names, f"build extra '{build}' must not list torchaudio"


def test_no_stale_leaf_extras() -> None:
    """The old torch<line>-<build> leaf extras must be gone (matrix collapsed to thin build extras)."""
    extras = _load_pyproject()["project"]["optional-dependencies"]
    stale = [name for name in extras if name.startswith("torch2")]
    assert not stale, f"stale leaf extras still present: {stale}"


def test_build_extras_routed_to_matching_index() -> None:
    """Each build extra routes torch AND torchvision to the wheel index that matches it."""
    sources = _load_pyproject()["tool"]["uv"]["sources"]
    for package in ROUTED_PACKAGES:
        routes = {(entry["extra"], entry["index"]) for entry in sources[package]}
        for build, index in BUILD_INDEX.items():
            assert (build, index) in routes, f"'{package}' for '{build}' not routed to {index}"


def test_lock_pairs_torch_and_torchvision_per_build() -> None:
    """For every build, torch and torchvision resolve from the matching index with the matching tag.

    A torchvision left on generic PyPI (the original cause of the torch/torchaudio CUDA-mismatch class
    of bug) has no ``+cuXXX`` entry from the build's index, so this asserts the consistent pair exists.
    """
    packages = _load_lock()["package"]
    for build, index_url in BUILD_INDEX_URL.items():
        for package in ROUTED_PACKAGES:
            matches = [
                p
                for p in packages
                if p["name"] == package
                and p.get("source", {}).get("registry") == index_url
                and _local_build_tag(p["version"]) == build
            ]
            assert matches, (
                f"no {package} entry in uv.lock tagged '+{build}' from {index_url}; "
                "torch and torchvision have drifted apart for this build"
            )


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


def test_horde_engine_requested_with_layerdiffuse_extra() -> None:
    """horde-engine must carry the [layerdiffuse] extra in the BASE deps, not an opt-in extra.

    The default Stable Diffusion pipeline wires in the layer_diffuse_apply / LayeredDiffusionApply node
    unconditionally, so that ComfyUI node must import for ANY job to run. horde-engine 3.2.4 demoted the
    node's sole requirement (diffusers) from a core dependency to the optional 'layerdiffuse' extra, so a
    plain horde_engine request silently stops installing diffusers and every inference faults with a
    KeyError on LayeredDiffusionApply. This guards against the extra being dropped on a future bump.
    """
    base_deps = _load_pyproject()["project"]["dependencies"]
    horde_engine_specs = [d for d in base_deps if _normalize(_dep_name(d)) == "horde-engine"]
    assert horde_engine_specs, "horde-engine is not declared in [project.dependencies]"
    extras = set().union(*(_requested_extras(d) for d in horde_engine_specs))
    assert "layerdiffuse" in extras, (
        "horde-engine must be requested as horde_engine[layerdiffuse] in [project.dependencies]; "
        "without it diffusers is not installed and the base pipeline's LayeredDiffusionApply node fails "
        "to import, faulting every job"
    )


def test_horde_engine_requested_with_upscale_extra() -> None:
    """horde-engine must carry the [upscale-extra] extra in the BASE deps, not an opt-in extra.

    upscale-extra pulls spandrel-extra-arches, a pure-Python universal wheel (no accelerator-specific
    native blocker) that broadens the upscaler architectures spandrel can load. It gates no feature
    on/off, so it rides in base on every backend rather than behind a per-backend feature extra. This
    guards against it being dropped on a future bump (which would silently narrow upscaler support).
    """
    base_deps = _load_pyproject()["project"]["dependencies"]
    horde_engine_specs = [d for d in base_deps if _normalize(_dep_name(d)) == "horde-engine"]
    assert horde_engine_specs, "horde-engine is not declared in [project.dependencies]"
    extras = set().union(*(_requested_extras(d) for d in horde_engine_specs))
    assert "upscale-extra" in extras, (
        "horde-engine must be requested as horde_engine[...,upscale-extra] in [project.dependencies]; "
        "without it spandrel-extra-arches is not installed and the extra upscaler architectures are "
        "unavailable on every backend"
    )


def test_lock_resolves_spandrel_extra_arches() -> None:
    """Require spandrel-extra-arches in uv.lock so `uv sync --locked` installs the extra upscaler arches.

    The pyproject request alone is not enough: update-runtime builds the runtime from the lock, so a lock
    regenerated without the upscale-extra extra would still leave spandrel-extra-arches uninstalled.
    """
    locked_names = {_normalize(p["name"]) for p in _load_lock()["package"]}
    assert "spandrel-extra-arches" in locked_names, (
        "spandrel-extra-arches missing from uv.lock — run 'uv lock' after adding the horde_engine[upscale-extra] extra"
    )


def test_lock_resolves_diffusers() -> None:
    """Require diffusers in uv.lock so `uv sync --locked` (embedded runtime) installs it.

    The pyproject request alone is not enough: update-runtime builds the runtime from the lock, so a lock
    regenerated without the layerdiffuse extra would still leave the layerdiffuse node un-importable.
    """
    locked_names = {p["name"] for p in _load_lock()["package"]}
    assert "diffusers" in locked_names, (
        "diffusers missing from uv.lock — run 'uv lock' after adding the horde_engine[layerdiffuse] extra"
    )
