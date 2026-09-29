"""Every numeric threshold in the system. No magic numbers in agent code.

Tune these against eval/run_eval.py. The starting values came from the brief;
the comments record what each one is measured against so retuning is informed.
"""

# --- card spend --------------------------------------------------------------
# A purchase is "large" at this multiple of the customer's baseline p95 amount.
LARGE_PURCHASE_MULT = 5

# --- inflows ----------------------------------------------------------------
# A non-salary credit counts as a large inflow at this fraction of one salary.
LARGE_INFLOW_SALARY_FRACTION = 0.5

# --- outflows ---------------------------------------------------------------
# An outbound transfer is "large" at this multiple of one salary.
LARGE_OUTFLOW_SALARY_MULT = 2
# A transfer this soon after a salary credit, for this fraction of it, is a sweep.
SWEEP_WINDOW_HOURS = 24
SWEEP_FRACTION = 0.8

# --- income -----------------------------------------------------------------
# Salary amount must move by more than this fraction to count as changed.
INCOME_CHANGE_FRACTION = 0.2
# Days past the expected salary date before we call it missing.
SALARY_GRACE_DAYS = 3

# --- engagement -------------------------------------------------------------
# Observed/expected activity ratio bands. Lower ratio = worse engagement.
ENGAGEMENT_DROP_MED = 0.5
ENGAGEMENT_DROP_HIGH = 0.25
# Trailing window for login/session comparison.
ENGAGEMENT_WINDOW_DAYS = 14

# --- card activity ----------------------------------------------------------
# Consecutive days with no purchase before we call card usage stopped.
CARD_SILENCE_DAYS = 7
# Only call it stopped if the baseline was genuinely active.
MIN_BASELINE_WEEKLY = 5
# Trailing window for purchase-rate comparison.
CARD_WINDOW_DAYS = 7

# --- standing instructions --------------------------------------------------
# Days past the expected day_of_month before an absent instruction is a signal.
SI_GRACE_DAYS = 3

# --- merchant mix -----------------------------------------------------------
# Report a category whose share of spend rose by more than this (absolute).
MERCHANT_SHIFT_DELTA = 0.10
# Trailing window for category-share comparison.
MERCHANT_WINDOW_DAYS = 30

# --- explanation suppression (support agent) --------------------------------
# How far back to look for a transaction the customer just explained.
SUPPRESSION_LOOKBACK_HOURS = 72
# A stated amount matches a transaction within this relative tolerance.
SUPPRESSION_AMOUNT_TOLERANCE = 0.10
# How long an explanation suppresses alarm about the transaction it explains.
SUPPRESSION_DURATION_HOURS = 72

# --- support agent memory context -------------------------------------------
# How far back to summarise this customer's own tickets for the prompt.
SUPPORT_HISTORY_DAYS = 30
