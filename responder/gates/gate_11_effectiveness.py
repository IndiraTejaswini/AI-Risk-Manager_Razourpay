from .types import Candidate, Context, GateResult, block_result, pass_result

NAME, VERSION = "effectiveness_below_impression_cost", "v1"
def gate(candidate: Candidate, context: Context) -> GateResult:
    if candidate.tier == "allow" or candidate.effectiveness * candidate.c_rto > candidate.impression_cost:
        return pass_result(NAME, VERSION)
    return block_result(NAME, VERSION, "intervention is not treatable")
