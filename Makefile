# Regenerate every table and figure in the README from raw data.
#
# ARCHITECTURE.md 11.3. The pipeline is strictly ordered: each step consumes the
# population the previous one defined, and several scripts assert their inputs against
# the committed numbers rather than trusting them.
#
# Nothing here is incremental. `make all` is a full regeneration, ~20 minutes, and the
# outputs are deterministic: two runs produce byte-identical artifacts apart from the
# generation timestamp each report stamps on itself.

PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest

.PHONY: all gate features model policy explain serving fairness expansion test docker clean help

help:
	@echo "make all      regenerate every eval/ artifact and figure from raw data"
	@echo "make test     run the test suite"
	@echo "make docker   build the scoring image"
	@echo "make clean    remove generated artifacts (not eval/, which is committed)"

all: gate features model policy explain serving fairness expansion
	@echo ""
	@echo "All artifacts regenerated. Reports in eval/, figures in eval/figures/."

# --- steps 0-2: population, labels, targets -----------------------------------------
gate:
	$(PYTHON) scripts/00_count_positives.py
	$(PYTHON) scripts/01_label_composition.py
	$(PYTHON) scripts/02_label_targets.py

# --- step 1: point-in-time feature matrix -------------------------------------------
features: gate
	$(PYTHON) scripts/03_feature_matrix.py

# --- step 3: model, and the significance testing that resolves its null --------------
model: features
	$(PYTHON) scripts/04_train_model.py
	$(PYTHON) scripts/05_significance.py
	$(PYTHON) scripts/06_join_audit.py

# --- step 4-5: calibration and the cost policy ---------------------------------------
policy: model
	$(PYTHON) scripts/07_calibrate.py
	$(PYTHON) scripts/08_policy.py

# --- step 6: SHAP and risk reasons ---------------------------------------------------
explain: policy
	$(PYTHON) scripts/09_reasons.py

# --- step 7: serving latency ---------------------------------------------------------
serving: explain
	$(PYTHON) scripts/10_latency.py

# --- step 8: fairness ----------------------------------------------------------------
fairness: serving
	$(PYTHON) scripts/11_fairness.py

# --- the feature ablation that did not resolve ---------------------------------------
# Trains two extra models and does not change the shipped one.  Kept in `all` because
# an ablation whose result is "no" still has to be reproducible.
expansion: fairness
	$(PYTHON) scripts/12_feature_expansion.py

test:
	$(PYTEST) -q -p no:cacheprovider

docker:
	docker build -t rto-detector .

clean:
	rm -rf artifacts/ .pytest_cache/
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
