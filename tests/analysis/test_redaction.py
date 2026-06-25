"""Unit tests for secret/identifier redaction (the safety core of the support bundle)."""

from __future__ import annotations

from horde_worker_regen.analysis.redaction import build_redactor

_API_KEY = "abcdEFGH1234ijklMNOP56"  # 22 chars, the shape of a real horde key
_CIVITAI = "cd92292204eaa0759418fdebc5ae6d79"


class TestSecretValueScrubbing:
    """The primary defense: the actual secret strings never survive, wherever they appear."""

    def test_scrubs_api_key_everywhere(self) -> None:
        """The key is removed from a config field and from an env-var echo in a traceback alike."""
        redactor = build_redactor(secrets=[_API_KEY])
        text = f"api_key: {_API_KEY}\n... AIHORDE_API_KEY={_API_KEY} in os.environ ..."
        scrubbed, count = redactor.scrub(text)
        assert _API_KEY not in scrubbed
        assert count >= 2

    def test_scrubs_civitai_token(self) -> None:
        """The CivitAI token is scrubbed by value."""
        redactor = build_redactor(secrets=[_CIVITAI])
        scrubbed, _ = redactor.scrub(f"civitai_api_token: {_CIVITAI}")
        assert _CIVITAI not in scrubbed

    def test_anonymous_key_is_not_treated_as_secret(self) -> None:
        """The published anonymous key (0000000000) is not scrubbed; it is not a secret."""
        redactor = build_redactor(secrets=["0000000000"])
        scrubbed, count = redactor.scrub("api_key: 0000000000")
        assert "0000000000" in scrubbed
        assert count == 0

    def test_idempotent(self) -> None:
        """Scrubbing already-scrubbed text changes nothing further."""
        redactor = build_redactor(secrets=[_API_KEY])
        once, _ = redactor.scrub(f"key={_API_KEY}")
        twice, count = redactor.scrub(once)
        assert twice == once
        assert count == 0


class TestPatternBackstop:
    """A foreign bundle whose config we never had: scrub the value after a known secret key anyway."""

    def test_redacts_unknown_value_after_known_key(self) -> None:
        """`api_key: <unknown>` is scrubbed even when that value was not in the secret set."""
        redactor = build_redactor(secrets=[])  # we do not know the value
        scrubbed, count = redactor.scrub("api_key: someUnknownKeyValue123")
        assert "someUnknownKeyValue123" not in scrubbed
        assert count == 1

    def test_does_not_touch_unrelated_keys(self) -> None:
        """A non-secret key=value (e.g. max_threads) is left intact."""
        redactor = build_redactor(secrets=[])
        scrubbed, _ = redactor.scrub("max_threads: 4")
        assert "max_threads: 4" in scrubbed


class TestNoEntropyScrub:
    """High-entropy but non-secret values (job ids, sha256, model names) must survive."""

    def test_sha256_and_job_id_untouched(self) -> None:
        """A sha256 checksum and a UUID job id are context, not secrets, and are preserved."""
        redactor = build_redactor(secrets=[_API_KEY])
        sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        job = "6e35ce70-44b8-41fd-8394-9bf158c053bf"
        scrubbed, count = redactor.scrub(f"sha256={sha} job={job}")
        assert sha in scrubbed
        assert job in scrubbed
        assert count == 0


class TestIdentifierRedaction:
    """Home path, username, and worker name: scrubbed when enabled, kept when not."""

    def test_scrubs_home_username_worker(self) -> None:
        """With identifier redaction on, all three are masked (both slash styles for the home path)."""
        redactor = build_redactor(
            home_path="C:\\Users\\Michael",
            username="Michael",
            worker_name="tazlin-tui-example",
            redact_identifiers=True,
        )
        text = "path C:\\Users\\Michael\\x and C:/Users/Michael/y, dreamer_name: tazlin-tui-example"
        scrubbed, count = redactor.scrub(text)
        assert "Michael" not in scrubbed
        assert "tazlin-tui-example" not in scrubbed
        assert "<HOME>" in scrubbed and "<WORKER_NAME>" in scrubbed
        assert count >= 3

    def test_identifiers_kept_when_disabled(self) -> None:
        """With identifier redaction off, paths/worker name survive (only secrets are scrubbed)."""
        redactor = build_redactor(
            secrets=[_API_KEY],
            home_path="C:\\Users\\Michael",
            worker_name="tazlin-tui-example",
            redact_identifiers=False,
        )
        scrubbed, _ = redactor.scrub(f"C:\\Users\\Michael dreamer: tazlin-tui-example key={_API_KEY}")
        assert "C:\\Users\\Michael" in scrubbed
        assert "tazlin-tui-example" in scrubbed
        assert _API_KEY not in scrubbed
