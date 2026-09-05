import hashlib
from .types import Candidate, Context, GateResult, block_result, pass_result

NAME, VERSION = "exploration_slice", "v1"
def gate(candidate: Candidate, context: Context) -> GateResult:
    arm = int(hashlib.sha256(candidate.decision_id.encode()).hexdigest(), 16) % 10000 < 200
    return block_result(NAME, VERSION, "exploration slice") if arm else pass_result(NAME, VERSION)
