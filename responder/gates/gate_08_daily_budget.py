from .types import Candidate, Context, GateResult, block_result, pass_result

NAME, VERSION = "merchant_daily_budget", "v1"
def gate(candidate: Candidate, context: Context) -> GateResult:
    return pass_result(NAME, VERSION) if context.merchant_daily_budget_available else block_result(NAME, VERSION, "merchant daily budget exhausted")
