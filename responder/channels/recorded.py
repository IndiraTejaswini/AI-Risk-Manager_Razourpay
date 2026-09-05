from responder.candidate import CandidateAction
from .base import validate


class RecordedChannel:
    def __init__(self):
        self.sent: list[CandidateAction] = []

    def send(self, action: CandidateAction) -> None:
        validate(action)
        self.sent.append(action)
