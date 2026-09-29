# Testing report

Everything here is from the committed code, run offline (`C360_LLM_OFFLINE=1`,
no API key). Every LLM step therefore takes its deterministic fallback. **These
are offline numbers; nothing in this report was measured with a live model.**

Reproduce:

```
C360_LLM_OFFLINE=1 python run_skeleton.py          # all three scenarios
C360_LLM_OFFLINE=1 python -m eval.run_eval all     # harness -> eval_results/
python test_skeleton.py && python -m tests.test_agents && python -m tests.test_decision_layer \
  && python -m eval.test_harness && python -m tests.test_categories && python -m tests.test_guardrails
```

The dataset folder `scenario_0N` does not share a number with its ground-truth
id: `scenario_01` = `scenario_05_major_medical_event`, `scenario_02` =
`scenario_06_new_child`, `scenario_03` = `scenario_07_churn_risk`.

## 1. Harness results (final)

| Scenario | state | band | action | hitl | timeliness | FP pass | overall |
|---|---|---|---|---|---|---|---|
| medical (05) | 100% | 33.3% | 66.7% | 0% | never | 100% | **0.6333** |
| new child (06) | 100% | 100% | 100% | 100% | on_time | 100% | **1.0000** |
| churn (07) | 100% | 33.3% | 100% | 100% | on_time | 100% | **0.8667** |
| **mean** | 100% | 55.6% | 88.9% | | | 100% | **0.8333** |

Stage 1 baseline for comparison: `0.4333 / 0.6250 / 0.4833`, mean 0.5139.
Every change between the two is logged in [`tuning_log.md`](tuning_log.md).

Actions that actually fired (all `escalated`; none executed):
- **New child:** `personalized_offer` / `childcare_savings_or_insurance_plan` from
  2026-03-26 to 04-08. Ground-truth subtype matches.
- **Churn:** `relationship_manager_escalation` /
  `premium_retention_offer_and_fee_waiver` from 2026-03-07, then the escalation
  follow-up `proactive_retention_outreach` later in the run. Ground-truth subtype
  matches.
- **Medical:** no action ever fires; the band never reaches high.

## 2. Failing checkpoints

| Scenario / as_of | Expected (state / band / action / hitl) | Actual | Top of scorecard | Cause |
|---|---|---|---|---|
| medical 03-12 | medical_hardship / medium / no_action / — | medical_hardship / **low** / no_action / auto_approved | medical 0.35 (merchant_shift); job_loss 0.30 (income_pattern); fraud 0.293 (large_purchase, outflow_pattern) | Working memory keeps one finding per key. The latest `income_pattern` is `salary_missing` (votes job_loss 0.30), which overwrote the earlier benefit-income reading (EVT_000406/428) that voted medical. The $8,500 City General Hospital bill (mcc `healthcare`) is scored as a `large_purchase`, i.e. fraud evidence. The medical evidence is split across states instead of adding up. |
| medical 03-26 | medical_hardship / high / support_intervention / escalated | medical_hardship / **medium** / **no_action** / **auto_approved** | medical 0.571 (merchant_shift, life_event_hint, intent_signal); job_loss 0.30; fraud 0.215 | 0.571 < 0.62. The explicit hardship ticket (EVT_000462) only moves the score from 0.559 (03-25) to 0.571. Also, 03-21 already scores 0.577, above 03-26, so any cutoff that makes 03-26 high fires on 03-21, outside the 1-day window ("early"). |
| churn 02-15 | churn_risk / low / no_action / — | churn_risk / **medium** / no_action / auto_approved | churn 0.545 (standing_instruction_change, relationship_damage, engagement_trend, sentiment_state) | February's standing instructions (rent, gym, utilities) are missing from the live stream in every scenario, and `standing_instruction_change` votes 0.30 churn for each absence. The detector is correct, and the data really does lack them. |
| churn 04-10 | churn_risk / high / proactive_retention_outreach / — | churn_risk / **medium** / proactive_retention_outreach / escalated | churn 0.58 | Evidence has halved over roughly 30 days, so the score is below 0.62. The action is right; the band is not. |

All other ground-truth checkpoints match on state, band, action and (where
specified) hitl: medical 02-15, new child 02-20 and 03-27, churn 03-08.

## 3. Red herrings

| Event | Scenario | Must not fire | Result |
|---|---|---|---|
| EVT_000382 tuition transfer | medical | compliance_fraud_hold, relationship_manager_escalation | PASS |
| EVT_000402 resort refund | medical | personalized_offer | PASS |
| EVT_000328 baby-monitor purchase | new child | compliance_fraud_hold | PASS |
| EVT_000447 tax refund | churn | personalized_offer | PASS |

Since calibration, actions really do fire in two scenarios, so these passes now
show the red herrings being ignored. In Stage 1 they passed only because nothing
ever fired. In the churn scenario, the tax refund does produce a
`wealth_growth_or_windfall` candidate (0.17–0.20), but it never leads.

## 4. Robustness

| Check | Result |
|---|---|
| Determinism: two full runs | byte-identical checkpoints, all 3 scenarios |
| `--perturb`: arrival shuffled ±24h + 5 duplicate events, seeds 1–10 × 3 scenarios | **30 / 30 byte-identical** to the normal run |
| Guardrails on vs off (`tests/test_guardrails.py`) | 0 fires, 0 vetoes, byte-identical checkpoints on all 3 |
| Injected legal threat (EVT_900001) | fires LEGAL_THREAT; immediate checkpoint at 2026-03-10T10:30Z: RM escalation / `legal_escalation_queue` / escalated; no customer message; no offer or outreach afterwards |
| Injected "power of attorney" (EVT_900002) | no fire |
| Injected fraud report (EVT_900003) | fires CUSTOMER_FRAUD_REPORT; `compliance_fraud_hold`, state `potential_fraud_or_takeover` |
| Masking: full names, `ACC_` in `out/logs/*` (daily, approvals, LLM trace, span trace) | 0 hits |

How strong the perturbation check is: the clock sorts events by `event_time`
before dispatch, so arrival order is absorbed by design. It is not a
watermarking test. Duplicates are dispatched twice and absorbed downstream
(idempotent `EventStore.add`, finding de-duplication).

Test suites (all exit 0):

| Suite | Checks |
|---|---|
| `test_skeleton.py` | 69 |
| `tests.test_agents` | 99 |
| `tests.test_decision_layer` | 71 passed, **2 soft failures**: churn 02-15 band (above) and "no churn-vs-windfall debate" (the windfall candidate never gets within the conflict margin) |
| `eval.test_harness` | 64 |
| `tests.test_categories` | 27 |
| `tests.test_guardrails` | 58 |

## 5. Bugs found and fixed

In Stage 1 verification (`eval_results/stage1_verification.md`):
1. **Duplicate final checkpoint.** `simulated_end` was written twice, so
   `as_of_time` did not strictly increase.
2. **`{scenario}_approvals.jsonl` missing** when no approval was requested, and
   never reset between runs, so request ids could collide.
3. **Customer's full name in the daily log**, via an unmasked counterparty name in
   event summaries.

During this pass:

4. **Category vocabulary.** The code listed `groceries` / `baby_supplies` and the
   data uses `grocery` / `baby_products`. This is now normalised stem matching
   (`config/categories.py`). The harness didn't change, because the old substring
   rule happened to catch `baby`, but three spurious unknown-category warnings are
   gone.
5. **Raw `account_id` in the daily log.** It is now `acct:<type>:<hash8>`.
6. **Masking regex eats ISO dates** (`2026-02-01` → `[PHONE_1]`). Worked around in
   span traces only; see the limitations below.

## 6. Known limitations

- **Synthetic policy corpus.** `policy/*.md` was written for this project. Subtypes
  and caps come from it, not from a real bank policy.
- **Offline fallbacks.** All numbers above use keyword/template/precedence
  fallbacks. The LLM paths (support classification, debate, review, drafting) have
  not been scored end to end with a live model. Their results are cached, so a live
  run would be reproducible, but it would differ from these numbers.
- **Category vocabulary.** Three stem groups (baby, health, grocery) cover what the
  three scenarios contain. An unseen category is logged and scored as ungrouped.
  The signal agent emits nothing for a newly created standing instruction, so the
  `daycare_payment` SI (EVT_000363) maps to the `baby` group but is never used as
  evidence.
- **Prompt-level masking.** Masking is regex over text (full name and name parts,
  `ACC_`, `CUST_`, emails, phones, 7+ digit runs) applied before LLM calls and log
  writes. It does not catch other identifying details a customer might write, such
  as addresses or relatives' names. Its phone pattern also over-matches dates in
  free text sent to the LLM.
- **Calibration fits three scenarios.** Three numbers were moved (`tuning_log.md`)
  with reasons that should generalise, but they were chosen by looking at this data.
  There is no held-out scenario.
- **Timing model.** Events are processed at their true `event_time`, even when
  they arrived late. A real stream would have to wait or revise.
- **No executor.** In `--auto` mode approvals stay `escalated`, and nothing is sent
  to a customer.
- **Guardrail hold lasts the whole run.** It can only be cleared by
  `hitl.clear_hold`, and the replay has no human.
- **`{scenario}_llm_trace.jsonl` is appended across runs**, not reset, so counts
  taken from it are cumulative. Per-run counts are printed by `run_skeleton.py`.
