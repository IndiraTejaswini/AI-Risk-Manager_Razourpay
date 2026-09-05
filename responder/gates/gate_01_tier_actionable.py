from .types import Candidate, Context, GateResult, block_result, pass_result

NAME, VERSION = "tier_is_actionable", "v1"
def gate(candidate: Candidate, context: Context) -> GateResult:
    return pass_result(NAME, VERSION) if context.tier_actionable and candidate.tier != "allow" else block_result(NAME, VERSION, "tier is not actionable")
