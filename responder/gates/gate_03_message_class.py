from .types import Candidate, Context, GateResult, block_result, pass_result

NAME, VERSION = "message_class_matches_tier", "v1"
def gate(candidate: Candidate, context: Context) -> GateResult:
    expected = "service" if candidate.tier == "confirm" else "promotional"
    return pass_result(NAME, VERSION) if candidate.message_class == expected else block_result(NAME, VERSION, "message class does not match tier")
