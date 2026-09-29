# Agentic Customer 360

**Problem.** A bank sees a steady stream of events for each customer: card
payments, transfers, logins, support tickets, KYC updates. The task is to replay
that stream day by day, infer what is happening in the customer's life (a new
child, a medical hardship, preparing to leave the bank), decide how confident we
are and what the bank should do, and route the action to a human when needed.
Red herrings, such as a large one-off purchase or a tax refund, must not trigger
the wrong action.

**Solution.** A pipeline of small agents. A guardrail agent catches legal threats
and fraud reports first. A signal agent finds patterns in transactions, a support
agent reads customer messages with an LLM, a synthesis agent combines the
evidence into one state and confidence, and an action agent picks an action from
bank policy and sends it for human approval. Every decision cites its evidence,
personal data is masked, and every run is reproducible and traced. Run
`python run_skeleton.py`, then `python -m eval.run_eval`; details are in
[`docs/`](docs/).
