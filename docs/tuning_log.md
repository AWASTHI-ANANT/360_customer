# Tuning log

Rules for this pass: only the band cutoffs (`NO_EVENT_MAX`, `LOW_MAX`,
`MEDIUM_MAX` in `config/thresholds.py`) and individual weights in
`config/evidence_rules.py` may change. Cutoffs are preferred over weights, as few
numbers as possible change, and no rule may be added. Calibration stops at the
best result that doesn't break a low/medium checkpoint or a red-herring check.

All scores come from the offline run (`C360_LLM_OFFLINE=1`) scored by
`python -m eval.run_eval all`. The format is overall per scenario: medical
(scenario_01) / new child (scenario_02) / churn (scenario_03), then the mean.

## Starting point

`0.4333 / 0.6250 / 0.4833`, mean **0.5139**. No scenario ever reached `high`, so
the action table (every entry requires `high`) never fired. Every checkpoint was
`no_action / auto_approved`.

Daily top scores before tuning (from a replay that captures the synthesis
scorecard each day):

| Scenario | Must stay below high | Must be high |
|---|---|---|
| churn | max 0.573 before 03-05 (02-15 = 0.545, expected low) | 03-07 = 0.664, 03-08 = 0.654 (on-time window 03-05 to 03-08) |
| new child | max 0.419 before 03-25 | 03-26 = 0.52, 03-27 = 0.51 (window 03-25 to 03-27) |
| medical | — | 03-26 = 0.472, but 03-21 is already 0.483 (window 03-25 to 03-26) |

No single set of cutoffs satisfies all three rows. New child's 0.51 must be high,
while churn's 0.545 on 02-15 must be low.

## Changes

| # | Change | Medical | New child | Churn | Mean | Why this is justified beyond this data |
|---|---|---|---|---|---|---|
| 0 | baseline | 0.4333 | 0.6250 | 0.4833 | 0.5139 | — |
| 1 | `MEDIUM_MAX` 0.75 → **0.62** | 0.4333 | 0.6250 | **0.8667** | 0.6417 | Evidence weights are 0.10–0.45 and halve every 30 days, so 0.75 needed three or more fresh strong signals at once and was unreachable. 0.62 means roughly two fresh strong signals plus corroboration, which is what "act now" should mean in any scenario. |
| 2 | `kyc_change` (dependents increase) → new_child weight 0.45 → **0.60** | 0.4333 | **1.0000** | 0.8667 | 0.7667 | A customer-declared dependents increase is an authoritative fact, not a behavioural inference. It should outweigh any single spending signal and reach high given one corroborating signal. |
| 3 | `merchant_shift` (health group) → medical weight 0.20 → **0.35** | **0.6333** | 1.0000 | 0.8667 | **0.8333** | Health-category spend is the only direct medical-cost signal. It now carries the same weight as the other direct signals (SI cancellation 0.30, own-name exit transfer 0.35) rather than half of them. |

Three numbers changed in total; no rules were added. `NO_EVENT_MAX` and `LOW_MAX`
are unchanged.

Rejected on the way:
- **`MEDIUM_MAX` 0.65** gave churn the same result but left new child needing a
  0.65 KYC weight. 0.62 needs less.
- **KYC weight 0.55** peaked at 0.594 on 03-27, still below high.
- **Health weight 0.30** tied churn at 02-15 (0.300 vs 0.309), so the medical state
  still lost.
- **Raising `LOW_MAX` above 0.545** would fix churn 02-15 (expected low) but
  squeezes medium into a sliver just under 0.62. That fits this one checkpoint, so
  I didn't take it.

## Result after change 3

| Scenario | state | band | action | timeliness | FP | overall |
|---|---|---|---|---|---|---|
| medical | 100% | 33.3% | 66.7% | never | 100% | 0.6333 |
| new child | 100% | 100% | 100% | on_time (03-26) | 100% | 1.0000 |
| churn | 100% | 33.3% | 100% | on_time (03-07) | 100% | 0.8667 |

Red herrings EVT_000382, EVT_000402, EVT_000328 and EVT_000447 all still pass,
now with real actions firing elsewhere in each run.

## Known misses (not chased)

- **Churn 02-15: medium, expected low (0.545).** The score is inflated by
  `standing_instruction_change` (0.30) for February mortgage/rent/gym/utilities
  postings that are missing from the live stream in all three scenarios. The
  detector is right that they are absent. Lowering that weight would also weaken
  the real SI cancellation on 03-04 and lose churn's 03-08 high.
- **Churn 04-10: medium, expected high (0.58).** Evidence has decayed by then. The
  action is still `proactive_retention_outreach`, the escalation follow-up.
- **Medical 03-12: low, expected medium (0.35).** The state is now right.
- **Medical 03-26: medium, expected high (0.571).** Also, 03-21 already scores more
  than 03-26, so any setting that makes 03-26 high fires 5 days early (timeliness
  "early").
