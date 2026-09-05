from responder.candidate import CandidateAction
from .base import validate


class DryRunChannel:
    def send(self, action: CandidateAction) -> None:
        validate(action)
