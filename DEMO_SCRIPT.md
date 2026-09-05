# Demo script — 5 minutes, to camera

**Rehearsal note, read before the first take.** This script is tight at 5:00 — roughly 760
spoken words at 150 wpm, with no slack. If a section runs long, **cut from the detector
charts at 1:45**. Drop the PR curve first and the bootstrap chart second; the decision curve
earns its place at 3:00 anyway. **Never cut from the responder section or the fold defect.**
Those are the two moments that distinguish this from a leaderboard submission.

Every number below was verified against the committed artifacts and the running service
while this script was written. Where a figure is not stable, the script says so out loud
rather than quoting it.

The terminal-level companion — exact `curl` commands and their responses — is
[docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md). This file is the spoken version.

---

## Props and setup

**Start these before recording. The API cold start is 15.5 seconds and the UI needs it up.**

```bash
python -m uvicorn api.main:app --port 8000     # wait for /health to answer
python -m uvicorn ui.app:app --port 8501       # in a second terminal
```

Confirm before you hit record:

```bash
curl -s localhost:8000/health
# {"status":"ok","model_version":"rto-label_b-6c0f21786683"}
```

### Tabs, in the order you will need them

| # | Application | What is open | Used at |
|---:|---|---|---|
| 1 | VS Code | `eval/TIER1_LOCK.json`, scrolled to `targets.primary` | 0:00 |
| 2 | VS Code | `eval/figures/pr_primary.png` | 0:30 |
| 3 | VS Code | `eval/figures/bootstrap_ap.png` | 0:55 |
| 4 | VS Code | `eval/figures/decision_curve.png` | 1:20 and 3:00 |
| 5 | Browser | `localhost:8501`, scrolled to "One order we act on, one we do not" | 1:45 |
| 6 | Browser | same page, "The same chain, run against a declared context" | 2:20 |
| 7 | VS Code | `eval/policy_table.md` | 3:00 |
| 8 | VS Code | `tests/test_calibration_folds.py`, top of file | 4:00 |

**Pre-load every tab.** Nothing in this script has time for a page to render.

**Windows, recording from Git Bash:** if you are running the Docker variant, check the mount
landed before going live — `docker inspect <container> --format '{{json .Mounts}}'`. Git Bash
rewrites the `/root/...` path inside `-v` and the container silently falls back to
downloading. Prefix with `MSYS_NO_PATHCONV=1`.

---

## 0:00–0:30 — Hook

| Time | Screen | What to say | What to click |
|---|---|---|---|
| 0:00 | Tab 1 — `eval/TIER1_LOCK.json` | "This scores cash-on-delivery orders for return-to-origin risk before dispatch, on real Olist outcomes — orders that actually shipped and actually did or did not arrive. Every number I am about to show you is frozen in this lock file, and there is a test that fails if any of them move." | Nothing. Let the JSON sit on screen. |
| 0:15 | Same, highlight `operating_points` | "Here is the headline. The top 197 orders — the top one percent of a nineteen-thousand-order test window — contain six of the hundred and fifty-four failures. That is three point nine times the base rate, and the ninety-five percent Wilson interval runs from one point four to six point five percent, so it excludes the base rate of nought point seven eight." | Select the `top_1pct` block. |

**Do not say "state of the art" or "high accuracy."** The interval excluding the base rate is
a stronger and more specific claim than either, and a senior engineer will hear the
difference immediately.

---

## 0:30–1:45 — Detector evidence

Three charts. Not six. Each gets one sentence on what it shows and one on why it is honest.

| Time | Screen | What to say | What to click |
|---|---|---|---|
| 0:30 | Tab 2 — `pr_primary.png` | "Precision-recall against the prevalence baseline, at nought point seven eight percent positives. The honest detail is the note in the corner: the y-axis is clipped, because at this prevalence the curve spikes to precision one point zero on the very first ranked row — one lucky true positive at the top of the list. Scaling to that spike would squash the entire informative range into the bottom two percent of the axis and hide the baseline. So it is clipped, and it says so on the figure." | Open tab 2. Point at the clipping note bottom-right. |
| 0:55 | Tab 3 — `bootstrap_ap.png` | "Paired bootstrap on the average-precision difference. Two distributions, both confidence intervals marked, and the hard line at zero. The upper one is the model against the prevalence baseline: plus nought point nought nought nine three, interval nought point nought nought three four to nought point nought two seven nine, clear of zero. The lower one is the full feature matrix against order fields alone — and its interval crosses zero. That one does not resolve, and it is on the same chart rather than in a footnote." | Open tab 3. Trace the zero line with the cursor. |
| 1:20 | Tab 4 — `decision_curve.png` | "Realised cost against a treat-above threshold, with the four policies marked. The shaded band is the assumed cost of a return, swept plus and minus fifty percent — because that number is an assumption, and drawing a single line would hide that. The curve is nearly flat, and that is the finding: the detector is not yet strong enough for the threshold choice to be the interesting question." | Open tab 4. Point at the band, then at the four markers. |

**If you are running long, this is the section to cut.** Dropping tab 2 saves 25 seconds and
costs the least.

---

## 1:45–3:00 — The responder

| Time | Screen | What to say | What to click |
|---|---|---|---|
| 1:45 | Tab 5 — comparison panel | "Two orders through the same chain. On the left, one we act on — tier confirm. On the right, one we do not — tier allow, and it is refused at gate one, `tier_is_actionable`. That refusal is what the API actually returned, a four-two-two, not something this page drew. Of nineteen thousand six hundred and sixty-two orders, the cost policy selects thirty-four; the other nineteen thousand six hundred and twenty-eight leave at gate one and are never looked at again." | Scroll to the comparison panel. Point at gate 1 in both columns. |
| 2:10 | Same, gates 2–13 | "Notice the rows below gate one say 'not evaluated at score time' rather than 'passed'. The scoring endpoint does not run the gate chain, so this page has no outcome for those gates and it declines to invent one." | Point at the greyed rows. |
| 2:20 | Tab 6 — declared context | "So here is the chain actually executing. Same registry, same shipped gate functions, no reimplementation — against a context that is written down rather than measured, because Olist has no consent timestamps, no do-not-disturb status and no send windows. I change exactly one input. At twenty past two in the afternoon, all fourteen gates pass and the candidate reaches dispatch. At twenty to eleven at night, gate seven refuses — TRAI's daylight send window — and gates eight through thirteen are never reached." | Scroll to the declared-context panel. Point at the 14:20 column, then the 22:40 column. |
| 2:50 | Same, hold on gate 7 | "That ordering is deliberate. The compliance gates run before the economic ones, so the cost calculation at gate eleven never gets to vote on a message that is illegal to send." | Point at the TRAI basis labels on gates 4 through 7. |
| 3:00 | — | "And to be completely plain: nothing was sent to anyone. Olist carries no contact details. This is mechanism, not effect." | — |

**Say that last line.** Do not let it be inferred. A judge who has to work out for themselves
that no message was sent will assume you were hoping they would not.

---

## 3:00–4:00 — The cost argument

| Time | Screen | What to say | What to click |
|---|---|---|---|
| 3:00 | Tab 7 — `eval/policy_table.md` | "Four policies, measured on the same test rows. Intervene on nothing. Intervene on everything — which costs ten point two times more than doing nothing, because at nought point seven eight percent prevalence you pay the abandonment cost on nineteen and a half thousand good orders to influence a hundred and fifty-four bad ones. The hand-written rule a merchant would actually write, which loses money and has an ROC-AUC of nought point five zero one one. And the model plus cost policy, on thirty-four orders." | Open tab 7. Walk down the four rows. |
| 3:25 | Same, point at the header | "Two columns are deliberately missing: RTO cost avoided, and net. Both need an effectiveness prior, and we have not measured one — the counterfactual does not exist in observational data. Either column would turn a table of measured quantities into an assumed one while looking exactly as authoritative. So break-even effectiveness replaces them: what fraction of treated failures would have to be prevented for this to pay for itself. That is computed entirely from observed quantities. The absent column is the argument." | Point at the caption, then at the break-even column. |
| 3:45 | Tab 5 — back to the acted order | "And this is the order it acts on. Sixty-one reais of goods, a hundred and eighty-two reais of freight. That makes a return expensive and makes annoying the customer cheap, so the Elkan threshold for this order falls to nought point zero zero eight two — against nought point zero nine five six for the ordinary order beside it. Eleven times the difference in threshold, on about ten percent difference in risk. **It is actionable because the threshold moved, not because the risk was high.** That is a cost-sensitive policy rather than a risk cutoff, and this order is where you can see it." | Switch to tab 5. Point at the two threshold figures in the panel headers. |

---

## 4:00–4:40 — What we found

| Time | Screen | What to say | What to click |
|---|---|---|---|
| 4:00 | Tab 8 — `tests/test_calibration_folds.py`, docstring | "One thing we found, because it is the one that changed how we test. The cross-validation fold generator did not tile its index range — every fold after the first was shifted by its own index, leaving k minus one slots unassigned. The output array was allocated with `np.empty`, so those slots held uninitialised memory, and that memory entered a model-selection Brier score." | Open tab 8, docstring visible. |
| 4:18 | Same | "Here is why it matters. On the committed artifacts it moved a Brier cell by **two times ten to the minus seven** — invisible in a diff. On another run a slot held about ten to the seventy-third, the Brier came back as **ten to the one hundred and forty-three**, and it flipped which calibration window we shipped. There is no middle. A reviewer diffing artifacts would have seen nothing wrong nearly every time." | Nothing. Let the two numbers land. |
| 4:32 | Same | "We found it by re-running the pipeline three times and getting three different answers. Not from the test suite, which was green at three hundred and fifteen tests the whole time — no test asserted that the folds tile. One does now, and it fails on all fourteen parametrisations against the old code." | Scroll to `test_folds_tile_the_index_range_exactly`. |

**This is the moment a senior engineer decides you are serious.** Do not rush the
2e-7 / 1e+143 contrast — pause after each number.

---

## 4:40–5:00 — Close

| Time | Screen | What to say | What to click |
|---|---|---|---|
| 4:40 | Tab 1 — back to the lock | "What is measured: the ranking, the calibration, the cost of each policy on real outcomes, and the break-even effectiveness. What is not: any intervention effect. Nothing was sent, the effectiveness numbers are assumptions we sweep rather than measure, and a savings figure literally cannot be rendered — there is a test asserting it raises." | Switch to tab 1. |
| 4:52 | Same | "The one thing I would want you to remember: this system tells you where it is weak, and every number in it comes from a file you can re-run." | Hold on the lock. Stop. |

---

## Three likely judge questions, with prepared answers

**"Average precision is 0.0171. Isn't that terrible?"**

> On its own, yes — and that is why it is framed as context rather than the headline. Average
> precision integrates the entire curve, including the ninety-nine percent of orders this
> system never touches. The claim we actually make is about the top of the ranking, which is
> the only part the policy layer acts on: six of a hundred and fifty-four failures in the top
> 197 orders, at 3.9× the base rate, with a Wilson interval that excludes the base rate, and a
> permutation p of nought point nought nought nought six against random ranking. We also report the
> things that do not resolve — the feature ablation's interval crosses zero, and so does the
> policy's cost saving.

**"You only treat 34 orders and catch 2 failures. What use is that?"**

> Correct, and we state it as a count rather than dressing it as a rate for exactly that
> reason. Two positives is not a sample — the interval on that precision runs from 1.6% to
> 19%. What the 34 demonstrates is that the per-order cost model does something a flat one
> cannot: with scalar costs the policy treats **zero** orders, because a single threshold sits
> above this model's 6.86% probability ceiling for every tier. The 34 are the orders whose
> individual economics work. Making that number larger is a detector problem, not a policy
> problem, and we have not pretended otherwise.

**"Is the responder actually wired up, or is this a mock?"**

> The gate chain, the state machine, the templates and the append-only log are real code with
> 192 tests, and what you saw executing was the shipped registry, not a reimplementation. Two
> things are honestly not connected. The scoring endpoint does not run the gate chain — which
> is why those rows read "not evaluated" rather than "passed" — and nothing in the serving
> path writes the outbox table, so the poller is built and tested but not reached. The context
> those gates ran against is declared rather than measured, and the panel is captioned to say
> so. Olist has no contact details, so the effect cannot be evaluated here at all; the three
> claims we make are that the mechanism is built, the effect is unvalidated, and the learning
> path is instrumented.

---

## Numbers you may quote, and one you may not

Verified against the committed artifacts and the running service.

| Quantity | Value | Source |
|---|---|---|
| Test window | 19,662 orders, 154 failures, 0.783% | `eval/TIER1_LOCK.json` |
| Top 1% | 197 orders, 6 of 154, 3.9×, CI [1.40%, 6.48%] | headline block |
| Treated set | 34 orders, 2 of 154 | lock, `eval/policy.md` §1 |
| Permutation p | 0.00060 | `eval/significance.md` §2 |
| Paired bootstrap | +0.0093, CI [+0.0034, +0.0279] | `eval/significance.md` §3 |
| Probability ceiling | 6.856% | `eval/calibration.md` §1 |
| Intervene on everything | 10.2× worse than nothing | `eval/policy.md` §3 |
| Hand rule ROC-AUC | 0.5011 | `eval/policy.md` §3 |
| Break-even, base | 5.98% | `eval/responder/breakeven.md` |
| Demo order threshold | 0.008244 vs 0.095588 | live `/score` |
| Test suite | 585 collected, 584 passed, 1 skipped | `python -m pytest` |
| Suite size when the fold defect was live | 315 | collected at that commit |
| Cold start | 15.5s | `eval/latency.md` §3 |

**Do not quote a latency verdict.** The same endpoint on the same machine has measured p50 at
64.9ms, 105ms, 125ms and 322ms across artifacts, and the instability is between process
invocations rather than within a run — three passes inside one run agree to within about one
percent, and the next run lands somewhere else entirely. If asked, say that: *"we measured it
five times and got roughly 90 to 325 milliseconds on unchanged code, the variation is between
process launches rather than request to request, so we report a range and dropped it from the
lock's assertions rather than pick a run and call it a verdict."*

**Do not quote the demo order's risk as a fixed number.** Four of its features derive from the
payload's `created_at` and the demo stamps that with the wall clock: four loads of the same
payload gave 0.012515, 0.012802, 0.013055 and 0.013850. The **threshold** held at 0.008243624
to nine decimal places in all four, and it is the figure the argument rests on anyway.

**Do not quote the secondary target's average precision of 0.2938.** It is inflated by a
join-cardinality artifact; corrected it is 0.0190 and below its own detection threshold. If it
comes up, that is a good story — it is in
[BUILD_CHALLENGES.md](BUILD_CHALLENGES.md) incident 9 — but the number is not a result.
