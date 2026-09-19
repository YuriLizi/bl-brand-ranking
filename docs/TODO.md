# Deferred work

Things consciously postponed, with enough context to pick them up cold. Ordered by how
much they'd strengthen the deliverable.

---

## 0. Run and verify the Docker stack (blocked on a BIOS setting)

**Status.** Docker Desktop 29.8.0 is installed. The engine **cannot start** on this
machine because firmware virtualization is off:

```
systeminfo → Virtualization Enabled In Firmware: No
```

The Ryzen 5 5600X supports it (`VM Monitor Mode Extensions: Yes`,
`Second Level Address Translation: Yes`) — it is simply disabled in the BIOS. WSL2 *and*
Hyper-V both require it, so there is no software workaround.

**Step 1 — BIOS (must be done by hand).**
Reboot → `Del` or `F2` at the splash screen → find **SVM Mode** (AMD's name for
virtualization; usually *Advanced → CPU Configuration*, sometimes *OC → CPU Features*) →
**Enable** → save and exit.

Confirm afterwards:

```powershell
systeminfo | Select-String "Virtualization Enabled In Firmware"
```

It must read `Yes`.

**Step 2 — Windows features (admin, one reboot).**

```powershell
wsl --install --no-distribution
```

**Step 3 — start Docker Desktop, then build and run.**

```powershell
docker compose --profile build build base   # ~15-25 min the first time
docker compose up -d mlflow
docker compose run --rm trainer
docker compose up -d serving scheduler
curl http://localhost:8000/health
```

**Already prepared, so the build should be close to correct:**

- `docker/Dockerfile.base` pinned to **python:3.12-slim** — it was 3.11, which would have
  failed on the researcher's PEP 701 f-strings. A `py_compile` step now fails the build
  immediately if that ever regresses.
- TabPFN weights and the NameDataset are baked at build time, and those steps are no
  longer `|| true` — a broken image fails at build rather than silently downloading
  weights at container start.
- `docker-compose.yml` bind-mounts `./mlflow_local:/mlflow`, so the containerised MLflow
  serves the **same four runs and two model versions** already recorded, instead of
  starting empty. This works because every stored artifact URI is `mlflow-artifacts:/...`
  (server-relative), never an absolute host path.
- Serving runs **1 worker** with a 6 GB limit, and `~/.wslconfig` caps WSL2 at 9 GB /
  6 CPUs. Each worker holds its own ~3.4 GB TabPFN context, so two workers plus MLflow
  will not fit in 16 GB of host RAM.

Docker Desktop has been removed from Windows startup so it does not run in the background
while this is deferred.

**Nothing else depends on this.** The venv path is fully verified and is what every
measurement in the README was taken from.

---

## 0b. Make the MLflow UI actually show something (deferred 2026-09-18)

**Symptom.** The runs carry 27 metrics but the UI has no meaningful charts — the metric
view is a list of single dots and the Compare view is just a table.

**Root cause, and it is a design error in `run.py`, not an MLflow limitation:**

```python
mlflow.log_metrics(capture.metrics)   # flat dict, no step=
```

Every metric is written as **one point with no step**, so there is no series to plot. And
data that is genuinely sequential was modelled as independent scalars:

| Logged today | Should be |
|---|---|
| `lead_accuracy_day1 … lead_accuracy_day7` (7 keys) | `lead_accuracy`, one key, `step=1..7` |
| `lead_weighted_f1_day1 … day7` (7 keys) | `lead_weighted_f1`, `step=1..7` |
| `payout_mape_thr5/10/15/20` (4 keys) | `payout_mape_by_threshold`, `step=5,10,15,20` |
| `payout_mae_thr5/10/15/20` (4 keys) | `payout_mae_by_threshold`, stepped |

Fixing just this turns 22 scalar keys into 4 charted series, gives per-day accuracy a real
trend line, and makes the Compare view overlay two runs instead of tabulating them.
Keep the `_overall` keys as scalars — those are legitimately single values.

**Then add artifacts the UI renders.** MLflow displays images and HTML inline, and there
is nothing of the sort today:

- **CatBoost feature importance** — `CB_bl_lead.get_feature_importance(prettified=True)`.
  Cheap, already computed, and the single most useful plot for a reviewer: it shows
  whether the model leans on survey answers or on traffic-source noise.
- **Confusion matrix** for the lead classifier, per day and overall.
- **Predicted vs actual payout** scatter, with the y=x reference — makes the 7.2% MAPE
  legible at a glance and exposes where the tail brands sit.
- **Per-day classification report as a table artifact** (CSV or HTML) rather than only
  buried in the researcher's `.log`.

**And add data lineage.** `mlflow.log_input(mlflow.data.from_pandas(...))` records the
dataset against the run, so the UI links a run to the data that produced it rather than
relying on the `delta_table_version` parameter alone.

**Where.** All of it goes in `src/bl_ranker/training/run.py` (metric stepping, artifact
logging) and `src/bl_ranker/training/metrics_capture.py` (emit `(name, value, step)`
triples instead of a flat dict). The researcher's evaluation code stays untouched — this
is all additive, same as the current parsing.

**Effort.** ~30–45 min. Use the `dataviz` guidance for the plots rather than default
matplotlib styling.

---

## 0c. Investigate what the lead classifier is actually learning

Surfaced immediately by the feature-importance chart added in 0b, on the first run that
produced one:

| feature | importance |
|---|---:|
| **client_name** | **41.0** |
| industry | 6.2 |
| city | 4.7 |
| session_day | 3.9 |
| from_start_to_register | 3.8 |

`client_name` -- *which brand is being scored* -- carries roughly 40% of total importance,
6.6x the next feature. Meanwhile `credit_score_num` does not appear in the top 20 at all,
and `monthly_revenue_num`, `loan_amount_num` and `time_in_business_num` sit near the
bottom.

**Why this matters.** The briefing's premise is that acceptance depends on the user's
profile: *"a business with $200k monthly revenue and a 720+ credit score is valuable to
some lenders and outside the criteria of others."* But the model appears to have mostly
learned each brand's **base acceptance rate** rather than user-to-brand matching. Since
inference cross-joins one user against all ten brands, the ranking may be close to a
static ordering (base rate x payout), only lightly modulated by the survey answers -- which
would undercut the personalisation the product is built on.

**Caveats before anyone acts on this.** CatBoost's default importance
(PredictionValuesChange) tends to overstate high-cardinality categoricals, and brand
support is wildly skewed (~29k rows for `fundera / nerdwallet` against 2 for
`healthy paws`), so part of this is structural rather than behavioural.

**How to check it properly:**

1. Re-rank with SHAP (`model.get_feature_importance(type='ShapValues')`) rather than the
   default metric.
2. Ablate: train with `client_name` removed and compare per-brand accuracy. If the survey
   features carry little, the honest conclusion is that the ranking is mostly static.
3. Compare the produced ranking across genuinely different user profiles -- a 720+/$200k
   business versus a sub-550/$5k one. If the top brand rarely changes, that is the
   product answer regardless of what the importances say.

This is an observation about the researcher's model, not a defect introduced here, and it
is not a blocker. It is the single most useful question the new charts raise.

---

## 1. Measure the hosted TabPFN path (deferred 2026-09-18)

**Why it matters.** The headline claim is "77,000 ms → 1,067 ms", but 77,000 ms is the
*local naïve port*, not the researcher's actual design. Their design calls the hosted
Prior Labs API. So the comparison that actually answers *"was going local right?"* is
**hosted vs local-optimised**, and that has never been run.

Prior Labs serve from GPUs, so two round-trips could plausibly total 1–3 s. If hosted
lands near 800 ms, the honest claim becomes "local is more *controllable*", not "local is
faster". Better to find that out before an interviewer asks.

**How to run it.** The switch already exists — no code changes needed:

```powershell
$env:USE_HOSTED_TABPFN = "true"     # TABPFN_TOKEN is already in .env
.venv\Scripts\python.exe -m uvicorn bl_ranker.serving.app:app --port 8089
.venv\Scripts\python.exe loadtest\simulate.py --url http://127.0.0.1:8089/rank --requests 20 --concurrency 1
```

Compare against the local numbers in `loadtest/results.json`. `/health` reports
`tabpfn_execution` so you can confirm which path is actually live.

**Before running it, note the data-governance point below** — it is arguably the stronger
argument anyway, and it does not depend on the benchmark.

---

## 2. Data governance: what the hosted path transmits

Not a task so much as a finding that deserves a slide of its own.

The researcher's design sends the TabPFN *context* to a third-party API **on every user
request**. That context is 1,000 rows of:

- user-derived features — `city`, `cellphone_prefix`, `gender`, `fname_len`, `lname_len`
- **and `payout` — the actual amount each brand pays per lead**

The second one is commercially sensitive: it is the margin structure of the business.
Running TabPFN locally removes that transmission entirely.

This argument stands on its own, independent of any latency measurement, and is probably
a better headline than the speed number. Worth raising with whoever owns the data policy.

---

## 3. Promotion gate on the weekly run

Today `--mode production` promotes to `@champion` unconditionally. Nothing prevents a bad
week becoming the serving model at 05:00 on Sunday with nobody watching.

Add a check: run the test-set evaluation, compare `payout_mape_overall` and
`lead_weighted_f1_overall` against the current champion's logged metrics, and refuse to
promote on a regression beyond a threshold. Roughly 20 lines in
`src/bl_ranker/training/run.py:_register`, and it is what makes the schedule genuinely
safe to leave unattended.

---

## 4. Strengthen the fidelity benchmark

`benchmarks/fidelity.json` uses 25 users, giving roughly ±16 pp on the top-1 agreement
figures — so the 80% vs 76% gap between `n_estimators` 4 and 2 is noise, not signal. The
headline result (52% at `n_estimators=1`) is far outside that band and is safe.

Re-run with ~150 users for tight intervals. Preprocessing dominates the runtime, so cache
the preprocessed frame to disk first or it is ~20 minutes per run.

---

## 5. GPU measurement

Every latency number here is CPU-only — the dev machine has an AMD card (no CUDA; ROCm is
Linux-only) and Databricks Free Edition serving is CPU-only. TabPFN is a transformer doing
a forward pass over a 1,000-row context, which is exactly the GPU-shaped workload.

The recommendation is stated in the README and the deck, but it is *reasoning*, not a
measurement. One run on any CUDA instance would settle it.

---

## 6. Ranking-quality monitoring

Latency and model version are logged; whether the ranking actually *earns more* is not
measured at all. MAPE on payout is a proxy, and not the one that matters — revenue per
session is.

The real move is a shadow/A-B setup: serve the champion, log a challenger's ranking
alongside, and compare on realised downstream payout.

---

## 7. Confidence is not expressed in the response

Brand coverage is extremely skewed — `fundera / nerdwallet` has ~29k labelled rows,
`credibly` has 11, `healthy paws` has 2. The ranking for the long tail rests on almost no
observations, and the response format cannot distinguish "confident" from "guessing".

Worth surfacing a confidence signal, or at minimum flagging it to whoever consumes the
ranking.

---

## 8. Report the `{day}` bug upstream

`bl_models_train.py:270` builds its per-day log message so that the segment containing
`{day}` is a plain concatenated string rather than an f-string, so every one of the seven
per-day accuracy reports logs the literal text `for day {day}`.

The briefing asks for that log to be kept *for accuracy assessment*, and as written it
cannot tell you which day is which. The fix is one character in the researcher's file —
which is why it was reported rather than applied here.
