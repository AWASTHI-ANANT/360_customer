# Stage 1 verification

Date: 2026-09-29. All runs offline: `C360_LLM_OFFLINE=1`, `ANTHROPIC_API_KEY` and
`ANTHROPIC_AUTH_TOKEN` unset, shared `ML_Core` Python 3.13.2 (no pytest installed; the
suites are plain-script runners). Commands below are run from the repo root.

**Scenario naming.** No folder is called `scenario_07`. The dataset folder
`scenario_03` contains ground truth `scenario_07_churn_risk`, and every
"scenario_07" check below runs on it. The other two folders:
`scenario_01` = `scenario_05_major_medical_event`, `scenario_02` = `scenario_06_new_child`.

**Paths.** The gates refer to `logs/` and `memory/`. The pipeline writes these under
`--out-dir`, which defaults to `out/`, so the gates check `out/logs/` and `out/memory/`.
Checkpoints go to `output/`.

## Part A: hard gates

| # | Gate | Result |
|---|------|--------|
| 1 | Full test suite, offline, no key | **PASS** (5 soft accuracy checks fail by design; see below) |
| 2 | Three scenarios end to end, no exceptions | **PASS** |
| 3 | Checkpoints / logs / memory present and valid | **PASS** after 2 bug fixes (initially FAIL) |
| 4 | Determinism: two runs byte-identical | **PASS** |
| 5 | Late / out-of-order + duplicate robustness (scenario_07) | **PASS** (`--perturb` added) |
| 6 | Masking: no full name / `ACC_` in LLM prompts, trace, approvals | **PASS** (a name leak in the daily log was also fixed) |
| 7 | Harness on all three; `baseline_v1.json` exists | **PASS** |
| 8 | Working tree committed | **PASS** (committed with this file) |

After the fixes, every gate from 1 to 7 was rerun from a clean `out/logs` and
`out/memory` in a single pass.

### Gate 1: test suite

```
python test_skeleton.py              exit=0   69 checks passed
python -m tests.test_agents          exit=0   99 checks passed
python -m tests.test_decision_layer  exit=0   68 checks passed, 5 failed
python -m eval.test_harness          exit=0   64 checks passed
```

`tests/test_decision_layer.py` has two kinds of check. A failed `hard()` aborts the
suite; none failed. `check()` compares against ground truth and never blocks; the
file labels these failures "tuning decisions for the user, not bugs". The 5 that
fail are the scenario_07 accuracy expectations: 02-15 band low (got medium),
03-08 band high (got medium), 03-08 action RM escalation (got no_action), 03-08
hitl escalated (got auto_approved), and no churn-vs-windfall debate episode. The
same misses appear in the harness results in Part B. Two hard checks were added
for the bugs fixed below (strictly increasing `as_of_time`, approvals log exists).

### Gate 2: end-to-end runs

`python run_skeleton.py` exited 0 with 0 callback errors and 0 tracebacks.

```
scenario_01  events 116/116 OK  days=74  findings=279  episodes=32  checkpoints=74
scenario_02  events 110/110 OK  days=74  findings=209  episodes=16  checkpoints=74
scenario_03  events 72/72 OK    days=74  findings=127  episodes=16  checkpoints=74
```

### Gate 3: outputs

The validator is independent of the pipeline and keeps its own copy of the README
enum lists. It checks: the file parses as a JSON array; all 7 fields are present;
every enum value is on the README lists; `as_of_time` strictly increases; there is
one checkpoint for each date from `simulated_start` to `simulated_end` (74 days,
2026-02-01 to 2026-04-15), with none missing and none outside that window; and
`out/logs/{s}_daily.jsonl`, `out/logs/{s}_approvals.jsonl`,
`out/memory/{cust}_working.json` and `_episodic.jsonl` all exist and parse.

**First run: FAIL**, two bugs:

1. **Duplicate final checkpoint.** Every file had 75 entries, with `2026-04-15T00:00:00Z`
   appearing twice, so `as_of_time` did not strictly increase. The clock already
   fires a day boundary for `simulated_end`'s date, and then `run_skeleton.py`
   unconditionally wrote another "final" checkpoint at `simulated_end`. Both copies
   came from the same hypothesis and decision, which only change at day boundaries.
   *Fix:* `run_skeleton.py` writes the final checkpoint only if the last one written
   is earlier than `simulated_end`.
2. **`{s}_approvals.jsonl` missing in all three scenarios.** `hitl.ApprovalLog` only
   created the file on the first request, and no action is ever proposed offline.
   The log was also never reset on a fresh run, so request ids (`…:0001`) would
   collide with an earlier run's.
   *Fix:* `ApprovalLog` / `hitl.configure` take `fresh`. A fresh run truncates the
   file (just as the daily log already does); otherwise the file is touched. An
   empty file records "zero approval requests".

**After the fixes: PASS.** Each scenario has 74 checkpoints, strictly increasing, with
no dates missing and all files present and valid.

### Gate 4: determinism

Two full offline runs, the second into a separate `--out-dir`:

```
scenario_01 byte-identical sha256=e631f67ae2076ae3…
scenario_02 byte-identical sha256=89d5307b4b8b622e…
scenario_03 byte-identical sha256=2cbe088168ebee90…
```

### Gate 5: late / out-of-order and duplicates

There was no injection mode, so `--perturb [--seed N]` was added to
`run_skeleton.py`, along with `--checkpoint-dir` so a perturbed run cannot
overwrite `output/`.
It jitters each live event's arrival by U(-24h, +24h), reorders the stream by
arrival, and sets `ingestion_time = max(arrival, event_time)`; `event_time` is
never touched. It then re-sends 5 randomly chosen events (same `event_id`), each
arriving up to 24h after the original. The default seed is 7.

```
python run_skeleton.py -s scenario_03 --perturb --out-dir <tmp> --checkpoint-dir <tmp>/output
  perturbed: arrival shuffled +/-24h (seed 7), duplicated EVT_000402, EVT_000436, EVT_000444, EVT_000449, EVT_000457
  replayed 77/72 events across 74 days
-> checkpoints byte-identical to the normal run
```

Extra evidence beyond the gate: seeds 1 to 10 on all three scenarios gave 30 of 30
byte-identical results.

How it holds up, so you can judge how strong this gate is:
- **The shuffle is absorbed by the clock.** `SimulatedClock._timeline` sorts
  everything by `(event_time, event_id)` before dispatch. That is the designed
  late-arrival mechanism: events are processed at their true time, and
  `EventStore.add` bisect-inserts. As a result, an event arriving after its day's
  boundary is still counted at that boundary, which a truly streaming system could
  not do. This gate verifies that property; it does not test watermarking.
- **Duplicates reach the agents.** Note "77/72 replayed": the clock dispatches both
  copies. `EventStore.add` ignores a re-ingested id, and the agents' finding
  signature de-dup absorbs the repeat, so the output does not change. However, the
  daily log records both copies, and with a live LLM a duplicated support ticket
  would be classified twice (a cache hit, so no cost or drift). I did not change
  this because the gate passes. The obvious hardening is to skip dispatch in
  `_on_event` when the id is already in the store.

### Gate 6: masking

I grepped `out/logs/*` and `out/llm_cache/*` (plus the gate 4 and gate 5 scratch
out-dirs) for "Marcus Vance", "Priya Sharma", "David Chen" (case-insensitive) and
`ACC_`:

| File(s) | name hits | `ACC_` hits |
|---|---|---|
| `{s}_llm_trace.jsonl` (all 3, 44 records incl. full `user_prompt`) | 0 | 0 |
| `{s}_approvals.jsonl` | 0 | 0 (empty; no requests) |
| `out/llm_cache/` | — | — (0 entries: offline never writes the cache) |
| `{s}_daily.jsonl` | 0 after fix (was 3) | 115 / 108 / 70 |

I also checked the LLM traces and approvals for first names, surnames and `CUST_`
on their own: 0 hits.

- **Leak fixed.** `scenario_03_daily.jsonl` contained "…outbound 22,500.00 USD to
  David Chen - Chase Bank…" 3 times (EVT_000461, 000469, 000471). The event summary
  quoted `counterparty_name` raw, and the customer was paying themself at another
  bank. That breaks `masking.py`'s own contract ("applied before … every log write
  of raw text"). *Fix:* `DailyLogWriter` now takes the profile and runs event
  summaries through `masking.mask()`. Checkpoints were byte-identical before and
  after the fix.
- **Not changed:** every remaining `ACC_` hit is the structured `account_id` field
  of daily-log event lines, an internal audit field. It falls outside the
  zero-tolerance set you named (LLM prompts, LLM trace, approval context), so I left
  it. Tell me if the daily log should mask it too.
- **Caveat:** offline, the LLM cache is empty and no approval request ever happens,
  so gate 6 only exercised the trace's `user_prompt` (written on the unavailable
  path) and an empty approvals file. `context_shown` masking will first be exercised
  by the Stage 2 injected-event tests.

### Gate 7: harness

`python -m eval.run_eval all` exited 0 and wrote `eval_results/scenario_0{1,2,3}_report.json`.
`eval_results/baseline_v1.json` exists. Its scores match the new run exactly, but its
`checkpoint_count` (75) and `checkpoint_sha256` values predate the duplicate fix and are
now stale. I left it untouched as the historical record, since Stage 2's regression
test no longer compares against it.

### Gate 8: committed

Committed on `main`; nothing was pushed. Local `main` and `origin/main` had already
diverged (3 local commits vs 1 remote) before this work.

### Changes made (all bug fixes; no weights, thresholds or tuning constants touched)

| File | Change |
|---|---|
| `run_skeleton.py` | Skip the duplicate final checkpoint; `--perturb`, `--seed`, `--checkpoint-dir`; pass `fresh` to HITL and `profile` to the daily log |
| `hitl.py` | `ApprovalLog(fresh=)`: file always exists, truncated on fresh runs |
| `c360/daily_log_writer.py` | Event summaries masked with `masking.mask()` |
| `tests/test_decision_layer.py` | 2 new hard checks (strictly increasing `as_of_time`, approvals log exists) |
| `output/*.json`, `eval_results/scenario_*_report.json` | Regenerated (only the duplicate final checkpoint removed; `checkpoints_submitted` 75 → 74) |

## Part B: soft results (report only)

### Harness summary

| Scenario (ground truth) | state | band | action | hitl | timeliness (on_time/early/late/never) | FP pass | overall |
|---|---|---|---|---|---|---|---|
| scenario_01 (05 medical) | 33.3% | 33.3% (dist 0.67) | 66.7% | 0% | 0 / 0 / 0 / 1 | 100% | 0.4333 |
| scenario_02 (06 new child) | 100% | 50.0% (dist 0.50) | 50.0% | 0% | 0 / 0 / 0 / 1 | 100% | 0.6250 |
| scenario_03 (07 churn) | 100% | 0.0% (dist 1.00) | 33.3% | 0% | 0 / 0 / 0 / 1 | 100% | 0.4833 |
| **mean** | 77.8% | 27.8% | 50.0% | 0% | 0 / 0 / 0 / 3 | 100% | **0.5139** |

Weights: state 0.30, band 0.20, action 0.25, timeliness 0.15, FP 0.10.

**The pattern behind most of the misses:** no scenario ever reaches `high`. Offline,
the top score peaks at 0.654 (churn, 03-08), 0.51 (new child, 03-27) and 0.472
(medical, 03-26), against `MEDIUM_MAX = 0.75`. The action agent never proposes
anything, so every checkpoint in all three scenarios is `no_action / auto_approved`.
That explains all of the hitl misses and all of the "never" timeliness verdicts.

### Failing checkpoints

scenario_01 (ground truth: scenario_05_major_medical_event)

| as_of | failed | expected state / band / action / hitl | actual | top 3 scorecard (score: strongest findings) |
|---|---|---|---|---|
| 2026-02-15 | state | medical_hardship / low / no_action / — | churn_risk / low / no_action / auto_approved | churn_risk 0.309: standing_instruction_change, engagement_trend<br>medical_hardship 0.200: merchant_shift<br>potential_fraud_or_takeover 0.120: outflow_pattern |
| 2026-03-12 | state, band | medical_hardship / medium / no_action / — | job_loss_or_income_disruption / low / no_action / auto_approved | job_loss_or_income_disruption 0.300: income_pattern<br>potential_fraud_or_takeover 0.293: large_purchase, outflow_pattern<br>medical_hardship 0.200: merchant_shift |
| 2026-03-26 | band, action, hitl | medical_hardship / high / support_intervention / escalated | medical_hardship / medium / no_action / auto_approved | medical_hardship 0.472: merchant_shift, life_event_hint, intent_signal<br>job_loss_or_income_disruption 0.300: income_pattern<br>potential_fraud_or_takeover 0.215: large_purchase, outflow_pattern |

scenario_02 (ground truth: scenario_06_new_child)

| as_of | failed | expected | actual | top 3 scorecard |
|---|---|---|---|---|
| 2026-03-27 | band, action, hitl | new_child_life_event / high / personalized_offer / escalated | new_child_life_event / medium / no_action / auto_approved | new_child_life_event 0.510: kyc_change, life_event_hint, income_pattern<br>churn_risk 0.092: standing_instruction_change<br>job_loss_or_income_disruption 0.049: income_pattern |

(2026-02-20 passes on state, band and action.)

scenario_03 (ground truth: scenario_07_churn_risk)

| as_of | failed | expected | actual | top 3 scorecard |
|---|---|---|---|---|
| 2026-02-15 | band | churn_risk / low / no_action / — | churn_risk / medium / no_action / auto_approved | churn_risk 0.545: standing_instruction_change, relationship_damage, engagement_trend, sentiment_state (only candidate) |
| 2026-03-08 | band, action, hitl | churn_risk / high / relationship_manager_escalation / escalated | churn_risk / medium / no_action / auto_approved | churn_risk 0.654: outflow_pattern, standing_instruction_change, engagement_trend, relationship_damage<br>wealth_growth_or_windfall 0.168: large_inflow |
| 2026-04-10 | band, action | churn_risk / high / proactive_retention_outreach / — | churn_risk / medium / no_action / auto_approved | churn_risk 0.580: standing_instruction_change, outflow_pattern, card_activity_trend, engagement_trend<br>wealth_growth_or_windfall 0.078: large_inflow |

The scorecard shown is the one the day's checkpoint was built from. I captured it by
wrapping `SynthesisAgent.on_day_boundary` in a scratch replay, and that replay's
checkpoints were byte-identical to `output/`. On "carried forward" days it is the
last scorecard actually computed.

### Red-herring checks

| Event | Scenario | Must not fire | Window | Result |
|---|---|---|---|---|
| EVT_000382 (tuition transfer) | scenario_01 | compliance_fraud_hold, relationship_manager_escalation | 02-05 10:15 → 02-08 10:15 (3 checkpoints) | PASS, none violating |
| EVT_000402 (resort refund) | scenario_01 | personalized_offer | 02-18 09:00 → 02-21 09:00 (3) | PASS, none violating |
| EVT_000328 (baby-monitor purchase) | scenario_02 | compliance_fraud_hold | 02-25 11:45 → 02-28 11:45 (3) | PASS, none violating |
| EVT_000447 (tax refund) | scenario_03 | personalized_offer | 02-28 09:00 → 03-03 09:00 (3) | PASS, none violating |

These passes are real in the harness's sense (not vacuous: there are checkpoints in
each window). But since no action is ever proposed anywhere, they don't yet show that
the red herrings are being *discriminated*. They will only mean something once bands
reach `high`.

### LLM fallbacks (offline; the reason in every case is `unavailable: C360_LLM_OFFLINE is set`)

| Step (`schema_name`) | scenario_01 | scenario_02 | scenario_03 | Fallback used |
|---|---|---|---|---|
| `support_analysis` (SupportAgent) | 2 | 2 | 2 | `keyword_fallback()` on the masked text |
| `synthesis_review` (SynthesisAgent rationale + band review) | 16 | 7 | 4 | `_fallback_rationale()`: state, band, top-3 findings with event ids; band unchanged |
| `debate` (SynthesisAgent conflict) | 11 | 0 | 0 | `_precedence_verdict()` (state precedence rules) |
| ActionAgent (draft / critique) | 0 calls | 0 | 0 | never reached: no action is ever proposed |

Every checkpoint that took a fallback path carries `[fallback]` in its notes.

### Observations (not fixed; your call)

- **MCC vocabulary mismatch.** `agents/signal_agent.py: KNOWN_MCC_CATEGORIES` has
  `groceries` and `baby_supplies`, but the data uses `grocery` (46 events) and
  `baby_products` (the scenario_02 signal purchase). This currently only logs a
  warning. I didn't trace whether it also weakens `merchant_shift` for new-child
  (child detection uses the substring token `baby`, so it may not). Aligning the
  names changes evidence, so I left it with the tuning work.
- **Tests write to the real `output/`.** `run_one()` in the tests writes
  `output/scenario_03_checkpoints.json` even when given a tmp out-dir. The content is
  deterministic, so this is harmless today; the new `checkpoint_dir=` parameter could
  point the tests at tmp instead.
- **Stale baseline.** See gate 7: `baseline_v1.json` has stale hashes and counts.
