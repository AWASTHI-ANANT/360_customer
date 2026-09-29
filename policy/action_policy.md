<!-- SYNTHETIC POLICY: authored for this project; the dataset ships no policy
     documents. Nothing here reflects a real bank's rules. -->

# Action Policy

## sensitive_inferences

These inferences concern a customer's private circumstances: a new child, a
medical hardship, a relocation, a marriage or relationship change, and general
financial distress or income loss.

- Any offer or intervention based on one of these inferences requires human
  approval before it reaches the customer. It is never auto-approved.
- The customer-facing message must not state or hint at the inference. Do not
  reference health, pregnancy or children, relationships, moving home, or money
  troubles. Describe the product or the help available, not the reason.
- The relationship-manager brief may state the inference in full, and must cite
  the event ids it rests on.

## cost_caps

Maximum monetary value of a retention credit or goodwill payment, by
customer_value_tier. These mirror config/thresholds.py RETENTION_CREDIT_CAP,
which is the enforced ceiling.

- low tier: 50
- mid tier: 150
- high tier: 500

Any draft naming a figure above the customer's tier cap fails critique and is
either revised or escalated.

## escalation

- An unresolved debate between two competing readings goes to a human.
- A hypothesis flagged needs_human goes to a human regardless of score.
- Any action other than no_action must not be auto_approved.
- A fraud or vulnerability hold is escalated, never actioned silently.
