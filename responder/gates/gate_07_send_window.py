from .types import Candidate, Context, GateResult, block_result, pass_result

NAME, VERSION = "send_window", "v1"
def gate(candidate: Candidate, context: Context) -> GateResult:
    return pass_result(NAME, VERSION) if candidate.message_class == "service" or context.send_window_open else block_result(NAME, VERSION, "outside send window")
