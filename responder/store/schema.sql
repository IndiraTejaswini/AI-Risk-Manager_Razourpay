CREATE TABLE IF NOT EXISTS inbox (
  event_key      TEXT PRIMARY KEY,
  account_id     TEXT NOT NULL,
  received_at    TEXT NOT NULL,
  response_json  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decision_log (
  seq                   INTEGER PRIMARY KEY AUTOINCREMENT,
  decision_id           TEXT NOT NULL,
  order_id              TEXT NOT NULL,
  account_id            TEXT NOT NULL,
  model_version         TEXT NOT NULL,
  policy_version        TEXT NOT NULL,
  cost_constants_id     TEXT NOT NULL,
  effectiveness_prior_id TEXT NOT NULL,
  gate_set_version      TEXT NOT NULL,
  template_version      TEXT,
  calibrated_p          REAL NOT NULL,
  tier                  TEXT NOT NULL,
  threshold_used        REAL NOT NULL,
  c_fp_impression       REAL NOT NULL,
  c_fp_triggered        REAL NOT NULL,
  c_fn                  REAL NOT NULL,
  top_reason_class      TEXT NOT NULL,
  reasons_json          TEXT NOT NULL,
  features_missing      INTEGER NOT NULL,
  state                 TEXT NOT NULL,
  state_reason          TEXT,
  actor                 TEXT NOT NULL,
  occurred_at           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_decision_current
  ON decision_log(decision_id, seq DESC);

CREATE TABLE IF NOT EXISTS action_outbox (
  decision_id     TEXT PRIMARY KEY,
  created_at      TEXT NOT NULL,
  claimed_until   TEXT,
  attempts        INTEGER NOT NULL DEFAULT 0,
  terminal        INTEGER NOT NULL DEFAULT 0
);

CREATE TRIGGER IF NOT EXISTS decision_log_no_update
BEFORE UPDATE ON decision_log
BEGIN
  SELECT RAISE(ABORT, 'decision_log is append-only');
END;

CREATE TRIGGER IF NOT EXISTS decision_log_no_delete
BEFORE DELETE ON decision_log
BEGIN
  SELECT RAISE(ABORT, 'decision_log is append-only');
END;

CREATE TRIGGER IF NOT EXISTS inbox_no_update
BEFORE UPDATE ON inbox
BEGIN
  SELECT RAISE(ABORT, 'inbox is append-only');
END;

CREATE TRIGGER IF NOT EXISTS inbox_no_delete
BEFORE DELETE ON inbox
BEGIN
  SELECT RAISE(ABORT, 'inbox is append-only');
END;
