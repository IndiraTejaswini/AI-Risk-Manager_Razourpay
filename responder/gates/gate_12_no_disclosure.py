import re
from .types import Candidate, Context, GateResult, block_result, pass_result

NAME, VERSION = "no_risk_disclosure", "v1"
def gate(candidate: Candidate, context: Context) -> GateResult:
    text = candidate.rendered_text.lower()
    forbidden = [candidate.reason_class, "risk"]
    forbidden.extend(("allow", "confirm", "prepaid_only", "defer", "block"))
    if re.search(r"\b\d+(?:\.\d+)?%?\b", text) and not (
        candidate.address and candidate.address.lower() in text
    ):
        return block_result(NAME, VERSION, "numeric risk disclosure")
    forbidden.extend(context.reason_vocabulary)
    leaked = next((token for token in forbidden if token and token.lower() in text), None)
    return block_result(NAME, VERSION, f"risk disclosure: {leaked}") if leaked else pass_result(NAME, VERSION)
