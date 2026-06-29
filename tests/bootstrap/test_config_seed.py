"""Unit tests for config seeding (copy the template once, never clobber)."""

from __future__ import annotations

from pathlib import Path

from worker_bootstrap import config_seed


def test_seeds_when_absent(tmp_path: Path) -> None:
    """A fresh install copies the template to bridgeData.yaml."""
    template = tmp_path / "bridgeData_template.yaml"
    target = tmp_path / "bridgeData.yaml"
    template.write_text("api_key: 0000000000\n", encoding="utf-8")
    assert config_seed.seed_config(template=template, target=target) is True
    assert target.read_text(encoding="utf-8") == "api_key: 0000000000\n"


def test_never_clobbers_existing(tmp_path: Path) -> None:
    """An existing bridgeData.yaml (the user's API key/worker name) is left untouched."""
    template = tmp_path / "bridgeData_template.yaml"
    target = tmp_path / "bridgeData.yaml"
    template.write_text("api_key: template\n", encoding="utf-8")
    target.write_text("api_key: MINE\n", encoding="utf-8")
    assert config_seed.seed_config(template=template, target=target) is False
    assert target.read_text(encoding="utf-8") == "api_key: MINE\n"


def test_noop_without_template(tmp_path: Path) -> None:
    """A missing template is a no-op rather than an error."""
    target = tmp_path / "bridgeData.yaml"
    assert config_seed.seed_config(template=tmp_path / "missing.yaml", target=target) is False
    assert not target.exists()


def test_cpu_install_enables_alchemist(tmp_path: Path) -> None:
    """A fresh CPU install flips the seeded alchemist flag on so the worker is useful out of the box."""
    template = tmp_path / "bridgeData_template.yaml"
    target = tmp_path / "bridgeData.yaml"
    template.write_text('api_key: 0000000000\nalchemist: false\nmodels_to_load:\n    - "top 2"\n', encoding="utf-8")

    assert config_seed.seed_config(template=template, target=target, backend_token="cpu") is True

    text = target.read_text(encoding="utf-8")
    assert "alchemist: true" in text
    assert "alchemist: false" not in text


def test_gpu_install_leaves_alchemist(tmp_path: Path) -> None:
    """A GPU install copies the template verbatim (alchemist stays as the template had it)."""
    template = tmp_path / "bridgeData_template.yaml"
    target = tmp_path / "bridgeData.yaml"
    template.write_text("alchemist: false\n", encoding="utf-8")

    assert config_seed.seed_config(template=template, target=target, backend_token="cu132") is True
    assert target.read_text(encoding="utf-8") == "alchemist: false\n"
