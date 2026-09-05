"""
FastAPI application.  ARCHITECTURE.md 11.1.

    POST /score   Razorpay webhook-shaped payload -> risk, tier, reasons

The service is built once on startup - model, calibrator, SHAP explainer, source tables
and the historical population - because the first request should not pay for the last
one's cold start, and because section 8 asks for the explainer to be cached.
"""

from __future__ import annotations

import sys
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

warnings.filterwarnings("ignore", message=".*LightGBM binary classifier.*")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, HTTPException, Request, status  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from api.contract import ScoreRequest, ScoreResponse  # noqa: E402
from api.service import LeakyPayloadError, ScoringService  # noqa: E402
from responder.ingest import ResponderIngest  # noqa: E402
from responder.explain.treatability import render_treatability  # noqa: E402
from responder.templates.registry import Tier as ActionTier  # noqa: E402
from responder.api.respond import build_router as build_respond_router  # noqa: E402

_state: dict[str, ScoringService | ResponderIngest] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["service"] = ScoringService()
    _state["ingest"] = ResponderIngest(_state["service"])
    yield
    _state["ingest"].close()
    _state.clear()


app = FastAPI(
    title="COD Return-to-Origin Risk Detector",
    description=(
        "Pre-shipment RTO risk scoring on a Razorpay webhook-shaped payload. "
        "Returns a calibrated P(fails | ships), an action tier, and the top three "
        "risk reasons. All monetary reasoning is in BRL - see policy/constants.py."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def service() -> ScoringService:
    if "service" not in _state:           # direct construction in tests / scripts
        _state["service"] = ScoringService()
    return _state["service"]


def ingest() -> ResponderIngest:
    if "ingest" not in _state:
        _state["ingest"] = ResponderIngest(service())
    return _state["ingest"]


app.include_router(build_respond_router(lambda: ingest().connection))


@app.exception_handler(LeakyPayloadError)
async def _leaky(request: Request, exc: LeakyPayloadError) -> JSONResponse:
    """
    A payload carrying a post-outcome field is a client error, not something to
    silently ignore. 422 with the offending fields named.
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "post_outcome_field_in_payload", "detail": str(exc)},
    )


@app.exception_handler(ValidationError)
async def _invalid(request: Request, exc: ValidationError) -> JSONResponse:
    """
    The body is validated by hand after the raw read (the leakage gate needs to see
    fields a typed model would drop), so Pydantic's error surfaces here rather than
    through FastAPI's own validation path. Without this handler an unsupported currency
    or a float amount would return 500 - a server error for what is squarely a client
    one.
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "invalid_payload",
            "detail": [
                {"field": ".".join(str(x) for x in e["loc"]), "message": e["msg"]}
                for e in exc.errors()
            ],
        },
    )


@app.get("/health")
def health() -> dict:
    s = service()
    return {"status": "ok", "model_version": s.model_version}


@app.post("/score", response_model=ScoreResponse)
async def score(request: Request) -> Response:
    """
    Score one order.

    The raw body is read before validation so the leakage gate sees fields that the
    typed model would otherwise drop - rejecting a post-outcome field requires noticing
    it, and a schema that ignores unknown keys would not.
    """
    raw = await request.json()
    service().assert_payload_clean(raw)
    stored = ingest().score(raw, request.headers.get("x-razorpay-event-id"))
    return Response(content=stored, media_type="application/json")


@app.get("/explain/{order_id}")
def explain(order_id: str) -> dict:
    decision = ingest().connection.execute(
        "SELECT decision_id, top_reason_class, tier FROM decision_log "
        "WHERE decision_id = ? ORDER BY seq DESC LIMIT 1",
        (order_id,),
    ).fetchone()
    if decision is not None:
        decision_id, top_reason_class, tier = decision
        order_row = ingest().connection.execute(
            "SELECT order_id, state, state_reason FROM decision_log "
            "WHERE decision_id = ? ORDER BY seq DESC LIMIT 1",
            (order_id,),
        ).fetchone()
        if tier not in {item.value for item in ActionTier}:
            raise HTTPException(status_code=422, detail="decision has no actionable tier")
        response = {
            "decision_id": decision_id,
            "top_reason_class": top_reason_class,
            "tier": tier,
            "epistemic_status": "ASSUMED",
            "treatability": render_treatability(top_reason_class, tier),
            "state": order_row[1],
            "state_reason": order_row[2],
            "gate_trace": [{
                "gate": "decision_state",
                "basis": "Operational",
                "passed": order_row[1] not in {"SUPPRESSED", "HELD_EXPLORATION"},
                "reason": order_row[2],
            }],
        }
        if order_row is not None:
            try:
                response["counterfactual"] = service().explain_order(order_row[0])
            except KeyError:
                response["counterfactual"] = None
        return response
    try:
        return service().explain_order(order_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="order not found") from exc
