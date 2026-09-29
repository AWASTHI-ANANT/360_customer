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


# ============================================================================
# Decision layer (synthesis -> action -> checkpoint)
# ============================================================================

# --- confidence banding -----------------------------------------------------
# Upper bounds on the synthesis score for each outcome. A score at or below
# NO_EVENT_MAX means nothing is happening; above MEDIUM_MAX is "high".
NO_EVENT_MAX = 0.15
LOW_MAX = 0.45
# 0.75 -> 0.62: with 30-day decay, 0.75 needed three
# fresh strong signals and was never reached, so no action could ever fire.
MEDIUM_MAX = 0.62

# Evidence decays with age: a finding's weight halves every this many
# simulated days, so a stale signal cannot hold a state up indefinitely.
EVIDENCE_HALF_LIFE_DAYS = 30

# --- conflicting hypotheses -------------------------------------------------
# Two candidate states conflict when the runner-up is within CONFLICT_MARGIN of
# the leader and itself scores at least CONFLICT_MIN_SCORE. That is what sends
# the decision to adjudication rather than taking the top score on faith.
CONFLICT_MARGIN = 0.15
CONFLICT_MIN_SCORE = 0.30

# --- action pacing ----------------------------------------------------------
# Do not fire the same action at a customer again inside this window.
ACTION_COOLDOWN_DAYS = 14
# How long an escalation stays open before a follow-up is due.
ESCALATION_FOLLOWUP_DAYS = 21

# --- action limits ----------------------------------------------------------
# Maximum monetary value of a retention credit, by customer_value_tier.
RETENTION_CREDIT_CAP = {"low": 50, "mid": 150, "high": 500}

# A fraud action is never taken on a weaker band than this.
FRAUD_ACTION_MIN_BAND = "medium"
