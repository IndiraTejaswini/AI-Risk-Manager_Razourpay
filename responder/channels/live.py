import os

from responder.candidate import CandidateAction
from .base import validate


class LiveChannel:
    def __init__(self, data_source: str):
        if os.environ.get("RESPONDER_LIVE") != "1":
            raise RuntimeError("live responder is disabled")
        if data_source.lower() == "olist":
            raise RuntimeError("live responder cannot use the Olist evaluation dataset")
        self.data_source = data_source

    def send(self, action: CandidateAction) -> None:
        validate(action)
        raise NotImplementedError("live transport is not configured")
