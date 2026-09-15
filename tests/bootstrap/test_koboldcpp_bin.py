"""Regression coverage for the pinned koboldcpp binary bootstrap."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from worker_bootstrap import detect, koboldcpp_bin

_HOST_PLATFORM = (
    koboldcpp_bin.KoboldcppPlatform.WINDOWS_X64 if os.name == "nt" else koboldcpp_bin.KoboldcppPlatform.LINUX_X64
)


def _stable_path(root: Path) -> Path:
    """Return the platform-specific published koboldcpp path under *root*."""
    return root / "bin" / ("koboldcpp.exe" if os.name == "nt" else "koboldcpp")


def _test_asset(payload: bytes, *, name: str = "koboldcpp-test-asset") -> koboldcpp_bin.KoboldcppAsset:
    """Return an asset whose pinned digest is the digest of *payload*, so verification stays genuine."""
    return koboldcpp_bin.KoboldcppAsset(
        host_platform=_HOST_PLATFORM,
        variant=koboldcpp_bin.KoboldcppVariant.CUDA,
        asset_name=name,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _fake_stream(payload: bytes) -> Callable[[str, Path], str]:
    """Return a stand-in for the streaming download that writes *payload* and hashes what it wrote."""

    def stream(url: str, destination: Path) -> str:
        destination.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    return stream


@pytest.mark.parametrize(
    ("os_name", "sys_platform", "machine", "expected"),
    [
        ("nt", "win32", "AMD64", koboldcpp_bin.KoboldcppPlatform.WINDOWS_X64),
        ("nt", "win32", "x86_64", koboldcpp_bin.KoboldcppPlatform.WINDOWS_X64),
        ("posix", "linux", "x86_64", koboldcpp_bin.KoboldcppPlatform.LINUX_X64),
        ("posix", "linux2", "AMD64", koboldcpp_bin.KoboldcppPlatform.LINUX_X64),
    ],
)
def test_supported_hosts_map_to_a_published_platform(
    os_name: str,
    sys_platform: str,
    machine: str,
    expected: koboldcpp_bin.KoboldcppPlatform,
) -> None:
    """Both spellings of x64 resolve, so a host is not declared unsupported over a machine string."""
    assert koboldcpp_bin._platform_for(os_name=os_name, sys_platform=sys_platform, machine=machine) == expected


@pytest.mark.parametrize(
    ("os_name", "sys_platform", "machine", "expected_in_message"),
    [
        ("posix", "darwin", "arm64", "darwin"),
        ("nt", "win32", "ARM64", "ARM64"),
        ("posix", "linux", "aarch64", "aarch64"),
        ("posix", "freebsd14", "x86_64", "freebsd14"),
    ],
)
def test_unsupported_host_names_what_it_found(
    os_name: str,
    sys_platform: str,
    machine: str,
    expected_in_message: str,
) -> None:
    """An unsupported host gets a specific error naming it, never a guessed asset."""
    with pytest.raises(koboldcpp_bin.UnsupportedKoboldcppPlatformError, match=expected_in_message):
        koboldcpp_bin._platform_for(os_name=os_name, sys_platform=sys_platform, machine=machine)


@pytest.mark.parametrize(
    ("host", "variant", "expected_name"),
    [
        (koboldcpp_bin.KoboldcppPlatform.WINDOWS_X64, koboldcpp_bin.KoboldcppVariant.CUDA, "koboldcpp.exe"),
        (koboldcpp_bin.KoboldcppPlatform.WINDOWS_X64, koboldcpp_bin.KoboldcppVariant.NOCUDA, "koboldcpp-nocuda.exe"),
        (koboldcpp_bin.KoboldcppPlatform.LINUX_X64, koboldcpp_bin.KoboldcppVariant.CUDA, "koboldcpp-linux-x64"),
        (
            koboldcpp_bin.KoboldcppPlatform.LINUX_X64,
            koboldcpp_bin.KoboldcppVariant.NOCUDA,
            "koboldcpp-linux-x64-nocuda",
        ),
    ],
)
def test_asset_selection_per_platform_and_variant(
    monkeypatch: pytest.MonkeyPatch,
    host: koboldcpp_bin.KoboldcppPlatform,
    variant: koboldcpp_bin.KoboldcppVariant,
    expected_name: str,
) -> None:
    """Every supported target selects the upstream asset published for it, at the pinned release."""
    monkeypatch.setattr(koboldcpp_bin, "host_platform", lambda: host)

    asset = koboldcpp_bin.koboldcpp_asset(variant=variant)

    assert asset.asset_name == expected_name
    assert asset.download_url.endswith(f"/{koboldcpp_bin.KOBOLDCPP_VERSION}/{expected_name}")
    assert len(asset.sha256) == 64


def test_every_supported_target_has_a_distinct_pinned_digest() -> None:
    """The pin table covers all four supported targets with no copy-paste digest between them."""
    targets = {
        (host, variant) for host in koboldcpp_bin.KoboldcppPlatform for variant in koboldcpp_bin.KoboldcppVariant
    }
    assert set(koboldcpp_bin._ASSETS_BY_TARGET) == targets
    digests = {asset.sha256 for asset in koboldcpp_bin._SUPPORTED_ASSETS}
    assert len(digests) == len(koboldcpp_bin._SUPPORTED_ASSETS)


@pytest.mark.parametrize(
    ("nvidia_present", "expected"),
    [(True, koboldcpp_bin.KoboldcppVariant.CUDA), (False, koboldcpp_bin.KoboldcppVariant.NOCUDA)],
)
def test_variant_follows_detected_hardware(
    monkeypatch: pytest.MonkeyPatch,
    nvidia_present: bool,
    expected: koboldcpp_bin.KoboldcppVariant,
) -> None:
    """A machine without an NVIDIA card gets the Vulkan/CPU build rather than the CUDA one."""
    monkeypatch.setattr(detect, "_nvidia_present", lambda: nvidia_present)

    assert koboldcpp_bin.default_variant() == expected


def test_verified_download_publishes_and_records_the_release(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A digest-matching download that reports the pinned version reaches the stable path."""
    payload = b"koboldcpp-one-file-binary"
    monkeypatch.setattr(koboldcpp_bin, "_stream_download_to", _fake_stream(payload))
    monkeypatch.setattr(koboldcpp_bin, "_reported_version", lambda executable: koboldcpp_bin.pinned_version_number())

    published = koboldcpp_bin._download_verified_koboldcpp(_test_asset(payload), tmp_path)

    assert published == _stable_path(tmp_path)
    assert published.read_bytes() == payload
    assert (tmp_path / "bin" / "koboldcpp-version").read_text(encoding="utf-8").strip() == (
        koboldcpp_bin.KOBOLDCPP_VERSION
    )
    assert koboldcpp_bin.koboldcpp_executable(tmp_path) == published
    assert {path.name for path in (tmp_path / "bin").iterdir()} == {published.name, "koboldcpp-version"}


def test_checksum_mismatch_never_replaces_the_published_binary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A corrupt or intercepted body leaves the last runnable binary and its recorded release intact."""
    stable = _stable_path(tmp_path)
    stable.parent.mkdir()
    stable.write_bytes(b"last-runnable-koboldcpp")
    stamp = tmp_path / "bin" / "koboldcpp-version"
    stamp.write_text("v0.001\n", encoding="utf-8")
    monkeypatch.setattr(koboldcpp_bin, "_stream_download_to", _fake_stream(b"corrupt"))

    with pytest.raises(koboldcpp_bin.KoboldcppProvisionError, match="failed SHA-256 verification"):
        koboldcpp_bin._download_verified_koboldcpp(_test_asset(b"the-real-asset"), tmp_path)

    assert stable.read_bytes() == b"last-runnable-koboldcpp"
    assert stamp.read_text(encoding="utf-8").strip() == "v0.001"
    assert {path.name for path in (tmp_path / "bin").iterdir()} == {stable.name, "koboldcpp-version"}


def test_failed_version_probe_publishes_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A binary that cannot report the pinned version is discarded instead of published."""
    stable = _stable_path(tmp_path)
    stable.parent.mkdir()
    stable.write_bytes(b"last-runnable-koboldcpp")
    payload = b"unrunnable-download"
    monkeypatch.setattr(koboldcpp_bin, "_stream_download_to", _fake_stream(payload))
    monkeypatch.setattr(koboldcpp_bin, "_reported_version", lambda executable: None)

    with pytest.raises(koboldcpp_bin.KoboldcppProvisionError, match="no readable version"):
        koboldcpp_bin._download_verified_koboldcpp(_test_asset(payload), tmp_path)

    assert stable.read_bytes() == b"last-runnable-koboldcpp"
    assert {path.name for path in (tmp_path / "bin").iterdir()} == {stable.name}


def test_wrong_reported_version_publishes_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A digest that matches a different release than the pin cannot pass as the pinned one."""
    payload = b"some-other-release"
    monkeypatch.setattr(koboldcpp_bin, "_stream_download_to", _fake_stream(payload))
    monkeypatch.setattr(koboldcpp_bin, "_reported_version", lambda executable: "0.001")

    with pytest.raises(koboldcpp_bin.KoboldcppProvisionError, match="reported 0.001"):
        koboldcpp_bin._download_verified_koboldcpp(_test_asset(payload), tmp_path)

    assert koboldcpp_bin.koboldcpp_executable(tmp_path) is None


def test_pinned_release_already_published_does_not_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Provisioning is idempotent: the recorded pin short-circuits the network and the probe."""
    stable = _stable_path(tmp_path)
    stable.parent.mkdir()
    stable.write_bytes(b"already-published")
    (tmp_path / "bin" / "koboldcpp-version").write_text(f"{koboldcpp_bin.KOBOLDCPP_VERSION}\n", encoding="utf-8")

    def refuse_download(asset: koboldcpp_bin.KoboldcppAsset, root: Path) -> Path:
        raise AssertionError("an already published pinned release must not be downloaded again")

    monkeypatch.setattr(koboldcpp_bin, "_download_verified_koboldcpp", refuse_download)

    assert koboldcpp_bin.ensure_koboldcpp(tmp_path) == stable


def test_published_binary_from_another_release_is_replaced(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A binary left by an older pin is re-provisioned rather than trusted as current."""
    stable = _stable_path(tmp_path)
    stable.parent.mkdir()
    stable.write_bytes(b"older-release")
    (tmp_path / "bin" / "koboldcpp-version").write_text("v0.001\n", encoding="utf-8")
    payload = b"pinned-release"
    monkeypatch.setattr(koboldcpp_bin, "koboldcpp_asset", lambda **kwargs: _test_asset(payload))
    monkeypatch.setattr(koboldcpp_bin, "_stream_download_to", _fake_stream(payload))
    monkeypatch.setattr(koboldcpp_bin, "_reported_version", lambda executable: koboldcpp_bin.pinned_version_number())

    assert koboldcpp_bin.ensure_koboldcpp(tmp_path) == stable
    assert stable.read_bytes() == payload


@pytest.mark.slow
def test_provisioned_binary_reports_the_pinned_version() -> None:
    """The binary this checkout actually provisioned runs and identifies itself as the pinned release."""
    published = koboldcpp_bin.koboldcpp_executable()
    if published is None:
        pytest.skip("koboldcpp is not provisioned in this install root")

    reported = koboldcpp_bin._reported_version(published)

    assert reported is not None
    assert koboldcpp_bin.pinned_version_number() in reported
