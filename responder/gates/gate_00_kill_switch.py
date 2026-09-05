from .types import Candidate, Context, GateResult, block_result, pass_result

NAME, VERSION = "kill_switch", "v1"
def gate(candidate: Candidate, context: Context) -> GateResult:
    return block_result(NAME, VERSION, "kill switch enabled") if context.kill_switch else pass_result(NAME, VERSION)
