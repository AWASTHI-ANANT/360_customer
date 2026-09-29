# Testing report

All results are offline (`C360_LLM_OFFLINE=1`, no API key), so every LLM step
used its deterministic fallback.

## Harness results

| Scenario | State | Band | Action | Timeliness | Red herrings | Overall |
|---|---|---|---|---|---|---|
| Medical hardship | 100% | 33% | 67% | missed | pass | 0.63 |
| New child | 100% | 100% | 100% | on time | pass | 1.00 |
| Churn risk | 100% | 33% | 100% | on time | pass | 0.87 |

Mean overall score: **0.83** (0.51 before calibrating three numbers).

## What still fails

- **Medical:** the confidence band stays one step too low (low instead of
  medium on 03-12, medium instead of high on 03-26), so no support action fires.
- **Churn:** the band is medium instead of low on 02-15. February's standing
  payments are missing from the data, which looks like early churn.
- **Churn:** the band is medium instead of high on 04-10, because old evidence
  fades. The action is still correct.

## Checks that pass

- Two runs give byte-identical output.
- Shuffling event arrival by ±24h and adding duplicate events gives the same
  output (30 of 30 runs).
- Guardrails: injected legal-threat and fraud messages fire; "power of attorney"
  does not; there are zero false fires on the real data.
- No customer names or account ids appear in any log.
- All test suites pass.

## Limitations

- The policy documents are written by us, not by a real bank.
- The live-LLM path has not been scored; these numbers are from fallbacks.
- Settings were tuned on only three scenarios, so they may overfit.
