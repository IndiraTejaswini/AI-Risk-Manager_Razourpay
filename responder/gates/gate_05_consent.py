from .types import Candidate, Context, GateResult, block_result, pass_result

NAME, VERSION = "consent_on_record", "v1"
def gate(candidate: Candidate, context: Context) -> GateResult:
    return pass_result(NAME, VERSION) if candidate.message_class == "service" or context.consent_timestamp else block_result(NAME, VERSION, "consent timestamp missing")
