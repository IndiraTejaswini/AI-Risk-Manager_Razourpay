"""Customer response endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from responder.states import State
from responder.transitions import TransitionError, transition

router = APIRouter()


class RespondRequest(BaseModel):
    decision_id: str
    response: str | bool | None = None
    confirmed: bool | None = None


def _target(response: str | bool | None, confirmed: bool | None) -> State:
    if response is None:
        if confirmed is None:
            raise HTTPException(status_code=422, detail="response must confirm or cancel")
        return State.CONFIRMED if confirmed else State.CANCELLED_AT_PROMPT
    if isinstance(response, bool):
        return State.CONFIRMED if response else State.CANCELLED_AT_PROMPT
    normalized = response.strip().lower()
    if normalized in {"confirm", "confirmed", "yes", "accept", "accepted"}:
        return State.CONFIRMED
    if normalized in {"cancel", "cancelled", "canceled", "no", "decline", "declined"}:
        return State.CANCELLED_AT_PROMPT
    raise HTTPException(status_code=422, detail="response must confirm or cancel")


def build_router(connection_provider):
    local_router = APIRouter()

    @local_router.post("/respond")
    def respond(payload: RespondRequest) -> dict:
        connection = connection_provider()
        row = connection.execute(
            "SELECT state FROM decision_log WHERE decision_id = ? "
            "ORDER BY seq DESC LIMIT 1",
            (payload.decision_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="decision not found")
        current = State(row[0])
        if current in {State.CONFIRMED, State.CANCELLED_AT_PROMPT}:
            return {"decision_id": payload.decision_id, "state": current.value}
        target = _target(payload.response, payload.confirmed)
        try:
            transition(connection, payload.decision_id, target, "respond", payload.response)
        except TransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"decision_id": payload.decision_id, "state": target.value}

    return local_router


@router.post("/respond")
def respond_without_store(payload: RespondRequest) -> dict:
    raise HTTPException(status_code=503, detail="responder store is not configured")
