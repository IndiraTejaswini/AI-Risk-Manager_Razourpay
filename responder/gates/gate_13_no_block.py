from .types import Candidate, Context, GateResult, block_result, pass_result

NAME, VERSION = "assert_no_block_tier", "v1"
def gate(candidate: Candidate, context: Context) -> GateResult:
    return block_result(NAME, VERSION, "block tier is forbidden") if candidate.tier == "block" else pass_result(NAME, VERSION)
