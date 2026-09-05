# API contract

`POST /score` — pre-shipment COD return-to-origin risk for one order.

ARCHITECTURE.md §11.1. **All monetary reasoning is in BRL**; see below.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/score` | Score one order |
| `GET` | `/health` | Liveness plus the current `model_version` |
| `GET` | `/docs` | OpenAPI UI, generated from the same models |

---

## Request

The payload is a **Razorpay webhook event envelope** wrapping Orders and Payments
entities, not a generic order dict. Integration is a webhook subscription rather than a
bespoke mapping.

Conventions follow the platform:

- entities nest under `payload.<name>.entity`
- **amounts are integers in the currency's smallest unit** (centavos). A float amount is
  rejected — it is a rounding bug waiting to happen
- `created_at` is Unix seconds
- `notes` is a free-form string map and is ignored by the scorer

### Currency

**`currency` must be `"BRL"`.** The model is trained on Brazilian orders and every cost
constant in `policy/constants.py` is BRL. A payload in another currency would be scored
against magnitudes that mean something different, and the cost model would silently price
the wrong one. The endpoint rejects it rather than converting behind the caller's back.

### Customer identity and cold start

`payload.customer.entity.customer_reference` is the merchant's stable customer key and is
what the point-in-time history is looked up on. **It is optional.** A guest checkout with
no customer block scores normally — on this panel 97% of orders have no prior history, so
cold start is the ordinary path, not an edge case.

### Fields that are rejected

A payload carrying a post-outcome field is **rejected with `422`**, not silently stripped:

```
order_status, status, order_delivered_carrier_date, order_delivered_customer_date,
delivered_at, shipped_at, order_approved_at, order_estimated_delivery_date,
review_score, label, label_a, label_b
```

The check walks the whole body, including nested objects and list elements. A client
replaying a historical order finds out immediately rather than receiving a plausible
score computed from data the model must never see. See
[`data/COLUMN_WHITELIST.md`](../data/COLUMN_WHITELIST.md).

---

## Response

| Field | Type | Meaning |
|---|---|---|
| `risk` | float | Calibrated **P(fails \| ships)** — conditional on the order shipping |
| `tier` | enum | `allow` \| `confirm` \| `prepaid_only` \| `defer` |
| `reasons` | list[str] | Top-3 templated SHAP reasons, merchant-facing |
| `model_version` | str | Derived from the booster and calibrator artifacts |
| `threshold_used` | float | Per-order Elkan p\* for the selected tier |
| `features_missing` | int | Count of null features for this order |

`risk` is **conditional on shipment**, not unconditional. An order that is never
dispatched cannot return to origin; unconditional risk is `P(ships) × risk` and this
service does not model `P(ships)`. See [`eval/label_targets.md`](../eval/label_targets.md).

`model_version` is a hash of the booster, the calibrator parameters and the feature
order, so any retrain changes it automatically and a response can always be traced to the
model that produced it. It is present on **every** response.

`threshold_used` is the per-order threshold for the tier that was selected. When the
decision is `allow` it reports the cheapest action tier's threshold — the bar the order
failed to clear.

**Nobody is blocked.** The worst outcome for a false positive is `prepaid_only`: being
asked to pay upfront (§7.5).

---

## Worked example

### Request

```json
{
  "entity": "event",
  "account_id": "acc_BFQ7uQEaa7j2z7",
  "event": "order.created",
  "contains": [
    "order",
    "payment",
    "customer"
  ],
  "created_at": 1532464897,
  "payload": {
    "order": {
      "entity": {
        "id": "53cdb2fc8bc7dce0b6741e2150273451",
        "entity": "order",
        "amount": 14146,
        "currency": "BRL",
        "receipt": "rcpt_00417",
        "created_at": 1532464897,
        "notes": {
          "channel": "web"
        }
      }
    },
    "payment": {
      "entity": {
        "entity": "payment",
        "amount": 14146,
        "currency": "BRL",
        "method": "boleto",
        "international": false,
        "emi_installments": 1
      }
    },
    "customer": {
      "entity": {
        "entity": "customer",
        "customer_reference": "af07308b275d755c9edb36a90c618231"
      }
    },
    "shipping_address": {
      "entity": {
        "zipcode": "47813",
        "city": "barreiras",
        "state": "BA",
        "country": "BR"
      }
    },
    "line_items": [
      {
        "product_id": "595fac2a385ac33a80bd5114aec74eb8",
        "seller_id": "289cdb325fb7e7f891c38608bf9e0962",
        "amount": 11870,
        "quantity": 1,
        "shipping_amount": 2276
      }
    ]
  }
}
```

### Response

```json
{
  "risk": 0.011832507605654664,
  "tier": "allow",
  "reasons": [
    "This product category has an elevated failure rate.",
    "Ordered in a period with an elevated failure rate.",
    "This order contains an unusual number of items."
  ],
  "model_version": "rto-label_b-6c0f21786683",
  "threshold_used": 0.06496706544866161,
  "features_missing": 3
}
```

---

## Running it

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t rto-detector .
docker run -p 8000:8000 rto-detector
```

**Verified**, not asserted — re-run against the current model after Task Q's PIT fixes
changed the model bytes:

| Check | Result |
|---|---|
| `docker build` | completes; layers unchanged since the last full build (deps are pinned and untouched) are cached, so this run built in seconds rather than the ~2m15s a genuinely first build pays for `apt-get` + `pip install` |
| `docker run` serves | yes |
| `GET /health` | 200, 16s after start (cache mounted correctly — see the Git Bash note below) |
| `POST /score` inside the container vs the host's batch prediction for the same order | 1.4e-17 difference (1e-12 required by `tests/test_api.py`) |
| Image size | 985MB |
| Cold start to first successful `/score` | ~16-25s, cache mounted; varies with host state the same way per-request latency does (`eval/latency.md`) |

### The dataset is a runtime prerequisite

**The image does not vendor the Olist dataset** — ~43MB of CSVs that belong to Kaggle,
and a container should not carry someone else's data. It is resolved at startup, in
order:

1. `OLIST_DATA_DIR` — a mounted directory of CSVs
2. a mounted `kagglehub` cache
3. a `kagglehub` download

**Verified behaviour with no cache present:** the container **downloads the dataset
anonymously** (~10s on a normal connection) and starts normally. No Kaggle credentials
are needed. It does not fail — but it does need **network access at runtime**, and that
is a prerequisite rather than a surprise at `docker run`.

To avoid the download, mount an existing cache:

```bash
docker run -p 8000:8000 \
  -v "$HOME/.cache/kagglehub:/root/.cache/kagglehub" \
  rto-detector
```

**On Windows, from Git Bash: verify the mount actually landed before trusting it.**
Git Bash auto-translates any argument that looks like a Unix path, and it does so
*inside* the `-v` flag: `/root/.cache/kagglehub` gets silently rewritten to a literal
Windows path before Docker ever sees it, and the container falls back to downloading —
no error, no warning, just a cache that was never mounted. `docker inspect <container>
--format '{{json .Mounts}}'` shows the `Destination` actually used; if it is not
`/root/.cache/kagglehub`, prefix the `docker run` with `MSYS_NO_PATHCONV=1` (or run it
from PowerShell/cmd, which do not rewrite the argument). Found verifying this repo's own
instructions — the download-fallback path is what actually ran the first time.

or point at CSVs you already have:

```bash
docker run -p 8000:8000 -e OLIST_DATA_DIR=/data \
  -v "/path/to/olist/csvs:/data" rto-detector
```

### Cold start

**~14 seconds** to the first successful `/score`. The container trains the model at
startup rather than loading a fitted artifact, and also builds the calibrator, the SHAP
explainer, the history store, and runs a store-vs-row-scan self-check on **every**
order in the historical population — not a sample; a fixed 40-order stride used to stand
in for this and passed on two genuinely defective implementations before it was widened
(`eval/feature_expansion.md` section 3.2).

Health checks should allow for it; the Dockerfile's `HEALTHCHECK` uses a 90s start
period. A production image would ship the fitted artifacts and load them, which removes
the training and the full-panel feature build from cold start.

---

## Latency

**No single latency figure is honestly citable on this hardware, so this section gives a
range and no verdict.**

Across five runs of unchanged code at the same commit, p50 landed between **~90ms and
~325ms** and p99 between **~100ms and ~375ms**. The budget is met when the host is quiet and missed when it is not.

The variation is **between invocations, not within them**. Within one run the three
passes typically agree to a few percent — one run's passes sat at 314.1 / 315.8 /
318.2ms, another's at 120.7 / 122.0 / 326.6ms. So re-running this measurement will not
tell you which regime you are in, and the worst-of-three protocol does not sample the
thing that varies. If you run it yourself you may well get a figure above 200ms; that is
the measurement being unstable, not the endpoint having changed.

| | Step 7 | Store, slice sums | Store, precomputed sums |
|---|---:|---:|---:|
| Total p50 | 3250.0ms | 327.9ms | **91–317ms across runs** |
| Feature construction p50 | 3216.5ms | 289.3ms | **81–319ms across runs** |

The two order-of-magnitude improvements are real and are not in doubt — they are far
larger than the noise band. What is not supportable is a point figure or a pass/fail
verdict.

The first improvement came from indexing the historical population once at startup
(`features/store.py`) instead of re-deriving the point-in-time aggregates from ~97k rows
per request. The second came from precomputing each group's cumulative sums instead of
summing its slice per query — and that was a **correctness** change, not an optimisation:
NumPy's pairwise summation disagreed with the batch path's sequential accumulation in the
last bits, and the startup self-check caught it. The speedup was a side effect.

The feature *definitions* are untouched throughout — only how the prior window is
evaluated — and both the startup self-check and the API-vs-batch parity test hold that
line. Full breakdown: [`eval/latency.md`](../eval/latency.md).
