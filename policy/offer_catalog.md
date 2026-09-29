<!-- SYNTHETIC POLICY: authored for this project; the dataset ships no policy
     documents. Nothing here is a real bank's product or pricing. -->

# Offer Catalogue

One section per inferred_state that may receive an action. Each bullet is an
allowed `action_subtype`. Eligibility and max_value are advisory; the hard cost
ceiling is config/thresholds.py RETENTION_CREDIT_CAP, enforced in code.

## churn_risk

- premium_retention_offer_and_fee_waiver: Waive the annual card fee and assign a
  named relationship manager. (eligibility: tier high, tenure >= 24 months,
  max_value: 500)
- fee_refund_goodwill_credit: One-off goodwill credit for a disputed fee.
  (eligibility: an unfavourably resolved fee dispute in the last 60 days,
  max_value: 150)
- relationship_review_call: Booked call to review products and pricing.
  (eligibility: any tier, max_value: 0)

## medical_hardship

- medical_hardship_payment_plan: Restructure repayments over a longer term with
  interest frozen. (eligibility: evidence of sustained medical cost or reduced
  income, max_value: 0)
- payment_holiday: Defer up to three monthly payments. (eligibility: accounts in
  good standing before the hardship, max_value: 0)
- fee_waiver_during_hardship: Suspend maintenance and late fees while the
  hardship plan is active. (eligibility: active hardship plan, max_value: 150)

## job_loss_or_income_disruption

- hardship_payment_plan: Reduced payments for an agreed period.
  (eligibility: income disruption evidenced by missing or reduced salary,
  max_value: 0)
- budgeting_support_call: Free session with a financial wellbeing specialist.
  (eligibility: any tier, max_value: 0)

## financial_distress_general

- hardship_payment_plan: Reduced payments for an agreed period.
  (eligibility: distress evidenced by balance or repayment behaviour,
  max_value: 0)
- budgeting_support_call: Free session with a financial wellbeing specialist.
  (eligibility: any tier, max_value: 0)

## new_child_life_event

- childcare_savings_or_insurance_plan: Savings product with a family insurance
  review. (eligibility: dependants increased, max_value: 0)
- education_savings_account: Long-horizon education savings wrapper.
  (eligibility: any tier, max_value: 0)
- family_insurance_review: Review of life and income protection cover.
  (eligibility: any tier, max_value: 0)

## marriage_or_relationship_change

- family_insurance_review: Review of life and income protection cover.
  (eligibility: any tier, max_value: 0)
- relationship_review_call: Booked call to review joint product options.
  (eligibility: any tier, max_value: 0)

## job_change_or_promotion

- high_yield_savings_offer: Preferential rate on a savings account.
  (eligibility: sustained income increase, max_value: 0)
- relationship_review_call: Booked call to review products and pricing.
  (eligibility: any tier, max_value: 0)

## relocation

- mortgage_or_home_loan_offer: Indicative home-loan terms and a broker referral.
  (eligibility: address change on file, max_value: 0)
- address_update_support: Assisted update of address across all products.
  (eligibility: any tier, max_value: 0)

## retirement_transition

- investment_advisory_session: Session with an investment adviser.
  (eligibility: any tier, max_value: 0)
- high_yield_savings_offer: Preferential rate on a savings account.
  (eligibility: any tier, max_value: 0)

## wealth_growth_or_windfall

- investment_advisory_session: Session with an investment adviser.
  (eligibility: sustained balance increase, max_value: 0)
- high_yield_savings_offer: Preferential rate on a savings account.
  (eligibility: any tier, max_value: 0)

## potential_fraud_or_takeover

- temporary_hold_pending_verification: Hold outbound payments until the customer
  is verified through a trusted channel. (eligibility: any tier, max_value: 0)

## elder_vulnerability_or_scam_risk

- protective_review_hold: Hold and route to the vulnerable-customer specialist
  team. (eligibility: any tier, max_value: 0)
