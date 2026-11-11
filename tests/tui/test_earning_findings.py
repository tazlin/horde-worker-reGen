"""The Insights tab's earning findings: which ``card_earning`` figures raise each one, and the copy it shows."""

from __future__ import annotations

from horde_worker_regen.analysis.finding_kinds import Finding, FindingKind, Severity
from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    CapacityWorkloadSettings,
    CardEarningCardRow,
    CardEarningRow,
    CardEarningSnapshot,
    CardEarningWindowRestart,
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.tui.recommendations import analyze
from horde_worker_regen.tui.widgets.earning import CAPACITY_SETTING_WORDS
from tests.analysis.test_finding_copy_style import _emit_problems

_NOW = 1_800_000_000.0
_EARNING_KINDS = {
    FindingKind.CLAIMED_MODEL_EARNS_BELOW_REST,
    FindingKind.MODEL_EARNS_BELOW_REST,
    FindingKind.CARD_EARNING_RESTARTED,
}


def _row(model: str, *, device_index: int = 0, **overrides: object) -> CardEarningRow:
    base: dict[str, object] = {
        "device_index": device_index,
        "model": model,
        "model_kudos_per_second": 0.42,
        "model_band_low": 0.36,
        "model_band_high": 0.48,
        "rest_kudos_per_second": 0.61,
        "rest_band_low": 0.55,
        "rest_band_high": 0.67,
        "jobs": 412,
        "model_buckets": 30,
        "rest_buckets": 34,
        "attributed_card_seconds": 18000.0,
        "claim_seconds": 12240.0,
        "kudos_gap_to_rest": 3420.0,
        "verdict": "below_the_rest",
    }
    base.update(overrides)
    return CardEarningRow(**base)  # type: ignore[arg-type]


def _card(device_index: int = 0, *, wall_seconds: float = 36000.0) -> CardEarningCardRow:
    return CardEarningCardRow(
        device_index=device_index,
        wall_seconds=wall_seconds,
        attributed_seconds=wall_seconds * 0.8,
        unattributed_seconds=wall_seconds * 0.2,
    )


def _earning(
    rows: list[CardEarningRow],
    *,
    cards: list[CardEarningCardRow] | None = None,
    **extra: object,
) -> CardEarningSnapshot:
    base: dict[str, object] = {
        "rows": rows,
        "cards": cards if cards is not None else [_card()],
        "window_started_at": _NOW - 9 * 3600,
        "verdict_min_jobs": 20,
    }
    base.update(extra)
    return CardEarningSnapshot(**base)  # type: ignore[arg-type]


def _snapshot(earning: CardEarningSnapshot, **config: object) -> WorkerStateSnapshot:
    summary: dict[str, object] = {"dreamer_name": "Earning", "worker_version": "12.0.0"}
    summary.update(config)
    return WorkerStateSnapshot(
        timestamp=_NOW,
        config=WorkerConfigSummary(**summary),  # type: ignore[arg-type]
        card_earning=earning,
    )


def _earning_findings(snapshot: WorkerStateSnapshot) -> list[Finding]:
    return [finding for finding in analyze(snapshot) if finding.kind in _EARNING_KINDS]


def _describe(finding: Finding) -> str:
    return "\n".join(
        [
            f"[{finding.badge}] {finding.title}",
            f"Headline: {finding.headline}",
            f"Do this: {finding.action}",
            f"Details: {finding.detail}",
            *(f"  - {line}" for line in finding.evidence),
        ],
    )


def test_a_claimed_model_below_the_rest_gets_a_card_with_its_figures() -> None:
    """A below-the-rest model with claim time is a suggestion that gives its claim share, rates, bands and window."""
    earning = _earning([_row("Flux.1-Schnell fp8 (Compact)"), _row("AlbedoBase XL (SDXL)", verdict="pays")])

    findings = _earning_findings(_snapshot(earning))
    print(*(_describe(finding) for finding in findings), sep="\n\n")

    assert [finding.kind for finding in findings] == [FindingKind.CLAIMED_MODEL_EARNS_BELOW_REST]
    finding = findings[0]
    assert finding.severity is Severity.SUGGESTION
    assert finding.headline == (
        '"Flux.1-Schnell fp8 (Compact)" earns below the rest of its card, and the worker asked the horde only for '
        "it for 34% of the card's time."
    )
    assert finding.action == (
        "No action needed if you hold the card for this model on purpose. To stop, set `whole_card_claim_policy` "
        "back to its default, and remove the model from `whole_card_claim_always_models` and from the model pool's "
        "pinned seats. The worker then asks the horde for your other models while it holds the card."
    )
    assert "the card did not take better-paid work for other models then" in finding.detail
    assert "`whole_card_claim_policy` is always" in finding.detail
    assert finding.evidence == [
        "It earns 0.42 kudos per second of card time. The rest of the card earns 0.61.",
        "It took 50% of the card's time.",
        "At the rest of the card's rate, that card time would have earned about 3,420 more kudos. This assumes "
        "the horde had other work for the card.",
        "Its 95% band is 0.36 to 0.48. The rest's band is 0.55 to 0.67.",
        "Counted from 412 of its jobs on the card.",
        "Measured over the last 9 hours.",
    ]


def test_a_measured_model_that_pays_or_is_unmeasured_gets_no_card() -> None:
    """A paying or unmeasured model gives no earning card, whatever its claim time or share."""
    earning = _earning(
        [
            _row("Paying", verdict="pays", model_kudos_per_second=0.70, attributed_card_seconds=30000.0),
            _row("Unmeasured", verdict="unmeasured", jobs=6, attributed_card_seconds=30000.0),
        ],
    )

    assert _earning_findings(_snapshot(earning)) == []


def test_two_claimed_models_below_the_rest_give_two_cards_most_claim_time_first() -> None:
    """Each below-the-rest model gets its own card, the one the worker asked for alone longest first."""
    earning = _earning(
        [
            _row("Shorter claim", claim_seconds=3600.0, attributed_card_seconds=20000.0),
            _row("Longer claim", claim_seconds=12240.0, attributed_card_seconds=9000.0),
        ],
    )

    findings = _earning_findings(_snapshot(earning))

    assert [finding.kind for finding in findings] == [FindingKind.CLAIMED_MODEL_EARNS_BELOW_REST] * 2
    assert [finding.headline.split('"')[1] for finding in findings] == ["Longer claim", "Shorter claim"]


def test_unclaimed_models_below_the_rest_are_notes_ordered_by_the_kudos_given_up() -> None:
    """Without claim time, each below-the-rest model is a note, whatever its share, the largest estimate first."""
    cards = [_card(0), _card(1)]
    small_share = _row(
        "Small share",
        device_index=0,
        claim_seconds=0.0,
        attributed_card_seconds=3600.0,
        kudos_gap_to_rest=684.0,
    )
    larger_gap = _row(
        "Larger gap",
        device_index=1,
        claim_seconds=0.0,
        attributed_card_seconds=20000.0,
        kudos_gap_to_rest=3800.0,
    )

    findings = _earning_findings(_snapshot(_earning([small_share, larger_gap], cards=cards)))
    print(*(_describe(finding) for finding in findings), sep="\n\n")

    assert [finding.kind for finding in findings] == [FindingKind.MODEL_EARNS_BELOW_REST] * 2
    assert [finding.headline.split('"')[1] for finding in findings] == ["Larger gap", "Small share"]
    assert findings[1].headline == '"Small share" earns below the rest of card 0 and took 10% of the card\'s time.'
    finding = findings[0]
    assert finding.severity is Severity.INFO
    assert finding.headline == '"Larger gap" earns below the rest of card 1 and took 56% of the card\'s time.'
    assert finding.action == (
        "No action needed if you serve this model on purpose. Otherwise remove it from your model list, or add it "
        "to `models_to_skip`, to stop serving it."
    )
    assert "did not ask the horde for this model alone" in finding.detail
    assert finding.evidence == [
        "It earns 0.42 kudos per second of card time. The rest of the card earns 0.61.",
        "At the rest of the card's rate, that card time would have earned about 3,800 more kudos. This assumes "
        "the horde had other work for the card.",
        "Its 95% band is 0.36 to 0.48. The rest's band is 0.55 to 0.67.",
        "Counted from 412 of its jobs on the card.",
        "Measured over the last 9 hours.",
    ]


def test_the_claimed_cards_come_before_the_unclaimed_note() -> None:
    """A claimed model's suggestion sorts ahead of an unclaimed model's note."""
    cards = [_card(0), _card(1)]
    earning = _earning(
        [
            _row("Unclaimed", device_index=0, claim_seconds=0.0, attributed_card_seconds=30000.0),
            _row("Claimed", device_index=1),
        ],
        cards=cards,
    )

    findings = _earning_findings(_snapshot(earning))

    assert [finding.kind for finding in findings] == [
        FindingKind.CLAIMED_MODEL_EARNS_BELOW_REST,
        FindingKind.MODEL_EARNS_BELOW_REST,
    ]


def test_a_row_on_a_card_without_wall_time_gives_no_card() -> None:
    """A share needs the card's wall, so a row whose card the snapshot does not list is left out."""
    earning = _earning([_row("No wall", device_index=3)])

    assert _earning_findings(_snapshot(earning)) == []


def test_an_empty_window_gives_no_card() -> None:
    """A snapshot with no earning figures and no restart gives no earning finding."""
    assert _earning_findings(_snapshot(CardEarningSnapshot())) == []


def test_a_restart_names_the_settings_that_changed() -> None:
    """A restart during the run names each changed setting by its config key, or in words where it has none."""
    restart = CardEarningWindowRestart(at=_NOW - 60, changed_settings=["max_threads", "num_models"])
    earning = CardEarningSnapshot(last_restart=restart)

    findings = _earning_findings(_snapshot(earning))
    print(*(_describe(finding) for finding in findings), sep="\n\n")

    assert [finding.kind for finding in findings] == [FindingKind.CARD_EARNING_RESTARTED]
    finding = findings[0]
    assert finding.severity is Severity.INFO
    assert finding.headline == (
        "The earning figures started again when `max_threads` and the number of models in `models_to_load` changed."
    )
    assert finding.action == "No action needed. This note clears once a model's earning is measured again."
    assert finding.evidence == []


def test_a_restart_found_at_load_says_the_settings_differed() -> None:
    """A restart found at startup has no list of settings, so the note says only that they differed."""
    earning = CardEarningSnapshot(last_restart=CardEarningWindowRestart(at=_NOW - 60, changed_settings=None))

    findings = _earning_findings(_snapshot(earning))

    assert [finding.headline for finding in findings] == [
        "The earning figures started again because the settings differed from the last run's.",
    ]


def test_the_restart_note_stays_while_every_verdict_is_unmeasured_and_clears_once_one_is_measured() -> None:
    """The note stands while the new figures have no verdict and clears as soon as any model is measured."""
    restart = CardEarningWindowRestart(at=_NOW - 600, changed_settings=["queue_size"])
    unmeasured = _row("New", verdict="unmeasured", jobs=6, claim_seconds=0.0)

    waiting = _earning_findings(_snapshot(_earning([unmeasured], last_restart=restart)))
    measured = _earning_findings(
        _snapshot(_earning([unmeasured, _row("Back", verdict="pays", claim_seconds=0.0)], last_restart=restart)),
    )

    assert [finding.kind for finding in waiting] == [FindingKind.CARD_EARNING_RESTARTED]
    assert measured == []


def test_the_restart_note_follows_the_stats_tab_for_a_worker_without_image_jobs() -> None:
    """A worker that does not serve image jobs gets the note only once it has figures, as the Stats tab shows it."""
    restart = CardEarningWindowRestart(at=_NOW - 60, changed_settings=["alchemy_forms"])

    empty = _earning_findings(_snapshot(CardEarningSnapshot(last_restart=restart), dreamer=False))
    with_rows = _earning_findings(
        _snapshot(
            _earning([_row("Some", verdict="unmeasured", claim_seconds=0.0)], last_restart=restart),
            dreamer=False,
        ),
    )

    assert empty == []
    assert [finding.headline for finding in with_rows] == ["The earning figures started again when `forms` changed."]


def test_every_capacity_setting_has_a_name_a_finding_can_show() -> None:
    """Each capacity and workload field is a config key or has words, so a restart never shows a bare field name."""
    unnamed = [
        field_name
        for field_name in CapacityWorkloadSettings.model_fields
        if field_name not in reGenBridgeData.model_fields and field_name not in CAPACITY_SETTING_WORDS
    ]

    assert unnamed == []
    assert set(CAPACITY_SETTING_WORDS) <= set(CapacityWorkloadSettings.model_fields)


def test_every_earning_finding_follows_the_copy_guide() -> None:
    """Each earning finding's headline and "Do this" line read as docs/how-to/write-a-finding.md asks."""
    cards = [_card(0), _card(1)]
    every_setting = list(CapacityWorkloadSettings.model_fields)
    snapshots = [
        _snapshot(_earning([_row("stable_diffusion_xl")])),
        _snapshot(
            _earning(
                [_row("Deliberate", device_index=1, claim_seconds=0.0, attributed_card_seconds=30000.0)],
                cards=cards,
            ),
        ),
        _snapshot(CardEarningSnapshot(last_restart=CardEarningWindowRestart(at=_NOW, changed_settings=["max_power"]))),
        _snapshot(CardEarningSnapshot(last_restart=CardEarningWindowRestart(at=_NOW, changed_settings=None))),
        _snapshot(
            CardEarningSnapshot(last_restart=CardEarningWindowRestart(at=_NOW, changed_settings=every_setting[:3])),
        ),
    ]
    findings = [finding for snapshot in snapshots for finding in _earning_findings(snapshot)]

    assert {finding.kind for finding in findings} == _EARNING_KINDS
    problems = [f"{finding.id}: {problem}" for finding in findings for problem in _emit_problems(finding)]
    assert not problems, "\n".join(problems)
