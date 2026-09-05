from .types import Candidate, Context, GateResult, block_result, pass_result

NAME, VERSION = "customer_fatigue", "v1"
def gate(candidate: Candidate, context: Context) -> GateResult:
    return block_result(NAME, VERSION, "customer fatigue limit reached") if context.customer_fatigued else pass_result(NAME, VERSION)
