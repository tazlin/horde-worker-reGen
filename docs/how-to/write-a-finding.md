# Write a finding

A finding is what the worker tells a person when something needs attention: in `horde-log diagnose`,
on the dashboard's Diagnostics tab, and in the live recommendations. The reader may run one card at
home, may not know the worker's internals, and may not read English as a first language. Write for
that reader. `tests/analysis/test_finding_copy_style.py` enforces the parts of this page that a test
can check.

## Two layers

Every finding has a plain layer and a detail layer.

The plain layer is what everyone sees:

- **Headline.** One sentence. What happened, with the numbers and names the detector measured. The
  detector writes it, because only the detector knows the numbers.
- **Do this.** One to three sentences telling the reader what to do. The kind's spec writes it, so
  the advice is the same every time the kind fires. When the fix depends on what was measured, the
  detector adds one sentence to the end. When there is nothing to do, say so: "No action needed."

The detail layer sits below, collapsed on the dashboard and printed in the CLI:

- **Detail.** What is going on underneath, in the same voice. Here you may name the subsystem
  (the scheduler, the VRAM budget, the safety check) and give the internal name once in brackets if
  someone might search for it. The spec writes it.
- The evidence lines, the `see also` cross-reference, and the reference page.

## Voice

- Say what happened, whether it matters, and what to do. In that order.
- Short sentences, up to 25 words in the plain layer and 30 in the detail layer. One idea per sentence.
- Plain words. "The worker stopped taking jobs", not "pop liveness was lost". If a plain word exists,
  the internal term is banned from the plain layer (the test carries the list).
- Name a config key when the reader has to change it, in backticks, exactly as it appears in the
  config file: `queue_size`. Nothing else goes in backticks except a `horde-log` command.
- No brackets, semicolons, dashes as punctuation, or "e.g." in the plain layer. Start a new sentence.
- Do not tell the reader what the code should do differently. That is a bug report, and it goes in an
  issue or a code comment, never in copy shown to an operator.
- Do not soften or pad. "This is fine" beats "This is admitted co-residency operating as intended".
- Do not talk down. The reader is capable; they just have not read the source.
- Numbers go in the headline, once. Do not repeat them in the advice.

## Severity

Pick the level by what the reader should do, and the badge says it for them:

| Level | Badge | Use when |
|-------|-------|----------|
| critical | Fix now | The worker is losing work or will be paused by the horde. |
| warning | Check | Something is wrong or wasteful and the reader should look. |
| suggestion | Try | Nothing is wrong. A change would likely earn more. |
| info | Note | Context the reader may want. No action. |

## Before and after

Before (plain layer, one sentence):

> Dedicated post-processing activity coincided with 3 child low-free-VRAM reading(s), but nothing
> corroborated a stall: no post-processing watchdog reap, no WDDM demand-paging, and no reading below
> the inference reserve. The scheduler admits sampling and post-processing co-residency when measured
> device truth affords it, so this is admitted co-residency operating as intended, not a stall.

After:

> Headline: Post-processing ran while free VRAM was low 3 times, and nothing stalled.
>
> Do this: No action needed.
>
> Detail: The worker lets post-processing share a card with image generation when the measured free
> memory allows it. A stall would show as a post-processing timeout, Windows paging GPU memory to RAM
> (WDDM demand paging), or free memory below the reserve kept for generation. None of those happened.
