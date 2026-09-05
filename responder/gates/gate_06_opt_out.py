from .types import Candidate, Context, GateResult, block_result, pass_result

NAME, VERSION = "opt_out_and_recontact_spacing", "v1"
def gate(candidate: Candidate, context: Context) -> GateResult:
    ok = candidate.message_class == "service" or (context.opt_out_keyword and context.recontact_allowed)
    return pass_result(NAME, VERSION) if ok else block_result(NAME, VERSION, "opt-out or recontact spacing missing")
