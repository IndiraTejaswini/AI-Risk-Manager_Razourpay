"""Webhook ingestion, deduplication, and decision-log persistence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

from api.contract import ScoreRequest
from responder.states import State
from responder.store.db import connect
from responder.transitions import start_decision


class ResponderIngest:
    def __init__(self, service):
        self.service = service
        configured = os.environ.get("RESPONDER_DB_PATH")
        self._temporary = configured is None
        self._path = Path(configured) if configured else Path(
            tempfile.mkdtemp(prefix="rto-responder-")
        ) / "responder.sqlite3"
        self.connection = connect(self._path)
        self._lock = Lock()

    @staticmethod
    def event_key(body: dict, header: str | None) -> tuple[str, str]:
        if header:
            return header, "event id header"
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), "canonical body hash"

    def close(self) -> None:
        self.connection.close()
        if self._temporary:
            self._path.unlink(missing_ok=True)
            self._path.parent.rmdir()

    def score(self, body: dict, event_id: str | None):
        key, derivation = self.event_key(body, event_id)
        with self._lock:
            existing = self.connection.execute(
                "SELECT response_json FROM inbox WHERE event_key = ?", (key,)
            ).fetchone()
            if existing is not None:
                return existing[0]

            request = ScoreRequest.model_validate(body)
            response, _ = self.service.score(request, body)
            decision_id = str(uuid4())
            response = response.model_copy(update={"decision_id": decision_id})
            response_json = json.dumps(
                response.model_dump(mode="json"), separators=(",", ":"), ensure_ascii=False
            )
            start_decision(
                self.connection,
                decision_id=decision_id,
                order_id=request.payload.order["entity"].id,
                account_id=request.account_id,
                response=response,
                state_reason=derivation,
            )
            with self.connection:
                self.connection.execute(
                    "INSERT INTO inbox(event_key, account_id, received_at, response_json) "
                    "VALUES (?, ?, ?, ?)",
                    (key, request.account_id, datetime.now(timezone.utc).isoformat(), response_json),
                )
            return response_json
