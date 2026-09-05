from .types import Candidate, Context, GateResult, block_result, pass_result

NAME, VERSION = "already_terminal", "v1"
def gate(candidate: Candidate, context: Context) -> GateResult:
    return block_result(NAME, VERSION, "decision is terminal") if context.terminal else pass_result(NAME, VERSION)
