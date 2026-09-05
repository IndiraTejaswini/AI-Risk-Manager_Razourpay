# Escalation rule

The responder allows exactly one retry escalation. `MAX_ESCALATIONS = 1` is
declared in `responder/states.py` and enforced by the transition writer, rather
than being a runtime configuration value.

The shape of this rule is informed by an external vendor report describing
roughly 7% of COD orders failing verification within 24 hours, split roughly
40%
abandoned, 30% fake or test, and 30% re-confirming on retry. This is an
unverified vendor figure, not a result of this project. It supports one retry
as a bounded recovery attempt; it does not establish those percentages here.
