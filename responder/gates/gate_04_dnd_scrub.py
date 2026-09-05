from .types import Candidate, Context, GateResult, block_result, pass_result

NAME, VERSION = "dnd_scrub", "v1"
def gate(candidate: Candidate, context: Context) -> GateResult:
    return pass_result(NAME, VERSION) if candidate.message_class == "service" or context.dnd_scrubbed else block_result(NAME, VERSION, "DND scrub missing")
