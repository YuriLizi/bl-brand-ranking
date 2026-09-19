# Runbook — running it locally and on Databricks

Step-by-step operating instructions. For what the system is, the architecture and the
measured results, see the [project README](../README.md).

> **Windows PowerShell:** use `curl.exe`, not `curl`. PowerShell aliases `curl` to
> `Invoke-WebRequest`, which rejects `-s`, `-H` and `-d`, and it uses a backtick rather
> than `\` for line continuation. Every `curl` command below works unchanged if you
> write `curl.exe` and keep it on one line.

Three ways to run it:

| | | |
|---|---|---|
| **Part A** | Docker |  needs only Docker, no Python setup |
| **Part B** | Local venv | if you would rather not use Docker; needs Python 3.12 |
| **Part C** | Databricks | optional; needs a workspace and ~20 minutes |

Parts A and B are independent -- either gives you the whole system. They share the same
MLflow store, so you can switch between them without losing runs or models.

In Part B the project interpreter is `.venv\Scripts\python.exe` (Python 3.12.10).

**Setting environment variables depends on your shell** — the commands below use
PowerShell, which is what the Windows terminal gives you by default:

| Shell | Syntax |
|---|---|
| PowerShell | `$env:PYTHONPATH = "src"` |
| cmd.exe | `set PYTHONPATH=src` |
| Git Bash | `export PYTHONPATH=src` |

Getting this wrong is the single most common way these steps fail: `set PYTHONPATH=src`
in PowerShell silently does nothing, and the next command dies with
`ModuleNotFoundError: No module named 'bl_ranker'`.

---

# Part A — Docker (recommended)

**This is the path to use.** It needs no Python, no venv, no dependency install -- only
Docker. Every measurement in the [project README](../README.md) was produced through it.

## A0. Requirements

Docker Desktop running. On Windows that means WSL2, which needs virtualization enabled in
firmware (`SVM Mode` on AMD, `VT-x` on Intel). Check with:

```powershell
systeminfo | Select-String "Virtualization Enabled In Firmware"
```

If it prints nothing, that usually means a hypervisor is already running -- which is the
success case. `Get-CimInstance Win32_ComputerSystem` showing `HypervisorPresent: True`
confirms it.

`~/.wslconfig` caps WSL2 memory. On a 16GB host, 9GB is a sensible ceiling: the serving
container holds a ~3 GB TabPFN context.

```
[wsl2]
memory=9GB
processors=6
```

## A1. Build (first time only, ~15-25 min)

```bash
docker compose --profile build --profile train build
```

`serving`, `trainer` and `scheduler` all derive from one base image, so they are built
together. Building only the base leaves any previously built image on the old one.

The base image is ~3.5GB. It bakes the TabPFN weights and the NameDataset in, so
containers never download them at start, and it `py_compile`s the researcher's scripts so
a wrong Python version fails the build instead of the first training run.

## A2. Start MLflow and the endpoint

```bash
docker compose up -d mlflow serving
```

MLflow comes up in seconds. **Serving takes ~130s** -- it loads `@champion` and fits the
TabPFN context once, which is the whole latency strategy. `docker compose ps` shows
`health: starting` until it is ready.

| | |
|---|---|
| MLflow | <http://localhost:5000> |
| Endpoint | <http://localhost:8088/docs> |

> **Port 8088, not 8000.** Port 8000 is very often taken on a developer machine, and a
> clash is confusing rather than obvious: Docker still reports the mapping and the
> container still reports healthy (its health check runs *inside* the container), but the
> other process answers your requests. Override with `SERVING_HOST_PORT=9000` if 8088 is
> busy too.

## A3. Rank a user

```bash
curl -s -X POST http://localhost:8088/rank -H "content-type: application/json" -d @examples/user.json
```

Returns the brands ranked by expected payout, with the serving model version and the
measured latency. Brands whose expected payout falls below $0.01 are filtered out by
the researcher's code, so fewer brands come back than the brand universe holds.

## A4. Train

```bash
docker compose run --rm trainer
```

Production mode: trains on all data, registers a new version, promotes it to `@champion`
and demotes the incumbent to `@previous`. **~3 min.**

Evaluation mode, which writes no artifacts by design:

```bash
docker compose run --rm trainer python -m bl_ranker.training.run --mode train_test
```

**~8 min**, 80% of it TabPFN scoring the held-out week. Afterwards the registry version
count must be **unchanged** -- that is the researcher's "evaluation registers nothing"
design, and it is the thing worth checking.

## A5. Pick up a new model

The service resolves `@champion` at startup, so a new version needs a restart:

```bash
docker compose restart serving
```

`/health` then reports the new `model_version`.

## A6. Roll back

```bash
docker compose run --rm trainer python -m bl_ranker.rollback --list
docker compose run --rm trainer python -m bl_ranker.rollback --to-previous
docker compose restart serving
```

`/health` reports the previous version. Roll forward with `--to-version N`.

## A7. Load test

Run from the host (needs the venv) or any machine with Python and `httpx`:

```bash
python -m loadtest.simulate --url http://localhost:8088/rank --requests 60 --concurrency 1 2 4
```

Expect p50 ~1.05-1.3 s at concurrency 1 and zero failures. Payloads are sampled from **real
rows** of the dataset rather than a replayed fixture, which is what caught a schema bug the
example payload passed straight through.

Two further tools cover the cases a single sweep cannot:

```bash
python -m loadtest.sweep          # concurrency 1 to 100 (~15 min)
python -m loadtest.burst --size 10 --gap 12 --duration 60   # bursts, open-loop
python -m loadtest.replay_demand  # replay real arrivals, no server needed
```

Methodology, raw results and charts: [../loadtest/README.md](../loadtest/README.md).

## A8. Scheduler

```bash
docker compose up -d scheduler
docker compose logs scheduler
```

Prints the next fire time, which should be the coming Sunday 05:00 UTC.

## A9. Stop

```bash
docker compose down
```

Data survives: the Delta table and the MLflow store are bind-mounted from the repo.

---

# Part B — Local venv (no Docker)

Same system, run directly on the host. Use this if you cannot run Docker, or when you want
to iterate on the code without rebuilding an image.

There is also `run.ps1`, a single entry point that sets `PYTHONPATH` and the tracking URI
for you -- getting those wrong is the most common way these steps fail:

```powershell
.\run.ps1 setup     # venv + dependencies, once
.\run.ps1 all       # ingest -> train -> serve -> one prediction
.\run.ps1 status    # what is running, what is registered
```

The steps below are what `run.ps1` does, spelled out.

## B1. What to look at in MLflow

Go to <http://127.0.0.1:5000> → experiment **`bl_brand_ranking`**.

**The runs.** Both modes appear, named `train_test-*` and `production-*`. Click any one.

- **Parameters tab** — the 13 technical parameters that make runs comparable:
  `mode`, `delta_table_version`, `input_rows`, `catboost_depth`, `catboost_n_estimators`,
  `catboost_eval_metric`, `catboost_task_type`, `tabpfn_context_size`, `tabpfn_execution`,
  `brand_rank_order`, `random_seed`, `days_for_test`, `git_sha`.
  `delta_table_version` records which version of the Delta table the run read, which is
  what makes a run reproducible from the *data* side and not only the model side.
- **Metrics tab** (on a `train_test` run) — 18 metrics. `lead_accuracy` and
  `lead_weighted_f1` as stepped series across the held-out week, their overall values, and
  `payout_mape`/`payout_mae` both overall and swept across thresholds.
  `lead_accuracy` reaches 1.00 on the last day because that bucket holds a single row;
  `lead_support` is logged alongside so the sample size is visible.
- **Artifacts tab** → `researcher_log/log_file_bl_train_*.log` — the researcher's own log,
  uploaded verbatim, with the full per-day classification reports.

**Compare two runs.** Tick both `train_test` runs → **Compare**. This is the point of
logging parameters as parameters: you get a side-by-side diff and can chart metrics
across runs. You cannot do that with a text file.

**The registry.** Top nav → **Models** → `bl_brand_ranker`. The newest version carries
the `@champion` alias and its predecessor carries `@previous`; those two aliases are what
serving resolves and what a rollback moves.

## B2. Rebuild from scratch (if needed)

Open three terminals in the repo root. **Terminal 1 — MLflow:**

```bash
.venv\Scripts\python.exe -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow_local/mlflow.db --artifacts-destination mlflow_local/artifacts --serve-artifacts
```

Leave it running. Check <http://127.0.0.1:5000> loads.

## B3. Load the data

**Terminal 2:**

```powershell
$env:PYTHONPATH = "src"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
.venv\Scripts\python.exe -m bl_ranker.data.ingest
```

Takes ~5 s. Prints `delta_version=<n>` (0 the first time; each re-ingest adds a version). Reads `Task/bl_full_data.csv`, writes the Delta
table under `data/delta/bl_sessions`.

## B4. Train

Evaluation mode — writes no artifacts, produces the accuracy report (**~8 minutes**):

```bash
.venv\Scripts\python.exe -m bl_ranker.training.run --mode train_test
```

Production mode — trains on all data and registers a version (**~2 minutes**):

```bash
.venv\Scripts\python.exe -m bl_ranker.training.run --mode production
```

Each prints its `run_id`. Refresh MLflow to see them.

> Both modes look idle for several minutes. That is CatBoost (800 trees) followed by
> TabPFN. There is no progress bar — the researcher's code logs to its file, not stdout.

## B5. Serve

**Terminal 3:**

```powershell
$env:PYTHONPATH = "src"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:TORCH_NUM_THREADS = "6"
.venv\Scripts\python.exe -m uvicorn bl_ranker.serving.app:app --host 127.0.0.1 --port 8088
```

**Startup takes ~130 seconds** — it loads the `@champion` model and fits TabPFN's cached
context. It is not hung. Wait for `Application startup complete`, then:

```bash
curl http://127.0.0.1:8088/health
```

Interactive API docs: <http://127.0.0.1:8088/docs> — requests can be sent from the
browser there. The service root, <http://127.0.0.1:8088/>, runs a ranking in place.

From the command line:

```bash
curl -s -X POST http://127.0.0.1:8088/rank -H "content-type: application/json" -d @examples/user.json
```

Returns the ranked brands with `rank`, `expected_payout`, the serving `model_version` and
the measured `latency_ms`.

## B6. Rollback

```bash
.venv\Scripts\python.exe -m bl_ranker.rollback --list
.venv\Scripts\python.exe -m bl_ranker.rollback --to-previous
```

Then restart the serving process — it resolves the alias at startup, so the restart is
the whole deployment step. Check `/health` and you will see `model_version` change.

Roll forward again with `--to-version N`.

## B7. Load test

Needs the endpoint running. Each request takes ~1 s, so keep the counts small:

```bash
.venv\Scripts\python.exe -m loadtest.simulate --url http://127.0.0.1:8088/rank --requests 20 --concurrency 1 2 4 8
```

Prints p50/p90/p95/p99 and throughput per level, and writes `loadtest/results/results.json`.

## B8. Tests

```bash
.venv\Scripts\python.exe -m pytest -q
```

22 tests, ~2 s.

---

# Part C — Databricks

> **Status.** This part has been executed on Databricks Free Edition: both jobs are
> deployed and have completed successfully. Job definitions, schedules and the run record
> are in [DATABRICKS.md](DATABRICKS.md). It is marked optional because it needs a
> workspace of your own; Parts A and B give the whole system without one.

## C0. What Free Edition can and cannot do

From the Free Edition limitations page:

- **Serverless compute only** — "custom compute configurations are not supported". The
  bundle therefore declares **no** `job_clusters` and **no** `node_type_id`; dependencies
  are declared per serverless `environment`. A bundle written the usual way with job
  clusters will simply fail here.
- **Max 5 concurrent job tasks.** Our jobs use 2 and 1. Fine.
- **Model Serving exists but is limited** — limited active endpoints, **no GPU serving**.
  Which means: the 1.07 s CPU latency from Part A is also what you would get there.
- **One 2X-Small SQL warehouse.**

## C1. Create the account

1. Go to <https://www.databricks.com/learn/free-edition> and sign up.
2. Note your workspace URL — `https://<something>.cloud.databricks.com`.
3. In the workspace: top-right avatar → **Settings** → **Developer** → **Access tokens**
   → **Manage** → **Generate new token**. Copy it; it is shown once.

## C2. Install and authenticate the CLI

```bash
winget install Databricks.DatabricksCLI
databricks --version
```

Then configure a profile:

```bash
databricks configure --host https://<your-workspace>.cloud.databricks.com --profile bl
```

Paste the token when prompted. Verify:

```bash
databricks current-user me --profile bl
```

> Store the token in the CLI profile, not in the repo. `.env` and `Task/tabpfn_sk.txt`
> are both gitignored, and no token should ever be committed.

## C3. Create the catalog and schema

In the workspace, open a SQL editor or a notebook and run:

```sql
CREATE SCHEMA IF NOT EXISTS workspace.bl;
```

**Free Edition has no `main` catalog** — its writable managed catalog is `workspace`, which
is the bundle's default. On a paid workspace with a `main` catalog, create it and the schema
there instead and pass `--var catalog=main` consistently from C6 onward.

## C4. Get the data into the workspace

The 50 MB CSV is not in git, so upload it once to a Unity Catalog volume:

```sql
CREATE VOLUME IF NOT EXISTS workspace.bl.raw;
```

Then **Catalog** → `workspace` → `bl` → **Volumes** → `raw` → **Upload to this volume**, and
upload `Task/bl_full_data.csv`. It lands at:

```
/Volumes/workspace/bl/raw/bl_full_data.csv
```

The ingest task reads `RAW_CSV_PATH`, so set it for the job — or, simpler for a one-off,
run the ingest yourself in a notebook cell:

```python
%pip install /Workspace/.../bl_ranker-1.0.0-py3-none-any.whl
import os
os.environ["RAW_CSV_PATH"] = "/Volumes/workspace/bl/raw/bl_full_data.csv"
os.environ["DELTA_TABLE_URI"] = "workspace.bl.bl_sessions"
from bl_ranker.data.ingest import ingest_csv_to_delta
print(ingest_csv_to_delta())
```

Either way you end up with the Unity Catalog table `workspace.bl.bl_sessions`, which is
what the training job reads.

## C5. Build the wheel

From the repo root:

```bash
.venv\Scripts\python.exe -m build --wheel --outdir dist
```

Produces `dist/bl_ranker-1.0.0-py3-none-any.whl`, which the bundle references.

## C6. Deploy and run

```bash
cd databricks
databricks bundle validate -t dev -p bl
databricks bundle deploy   -t dev -p bl
databricks bundle run bl_weekly_production_training -t dev -p bl
```

`validate` catches schema problems before anything is uploaded — run it first every time.

To use a different catalog or schema:

```bash
databricks bundle deploy -t dev -p bl --var catalog=main --var schema=bl
```

The default is `--var catalog=workspace`, which is what Free Edition provides.

## C7. What to look at in Databricks

- **Jobs & Pipelines** (in the left nav; this was called *Workflows* in older Databricks
  UI versions) → `[BL] Weekly production training`. You will see the two
  tasks (`ingest` → `train_production`) and, on the job detail page, the schedule showing
  **Sunday 05:00 UTC**.
- **Experiments** (left nav) → `bl_brand_ranking` → the same parameters, metrics and the
  researcher's log artifact as in Part A. The tracking code is identical; only the URI
  differs.
- **Catalog** → `workspace.bl.bl_brand_ranker` — the registered model in Unity Catalog,
  with the `@champion` alias. Under UC, model names are three-level
  (`catalog.schema.name`), which is why the bundle passes `--registered-model`.
- **Catalog** → `workspace.bl.bl_sessions` → **History** tab — the Delta versions, matching
  the `delta_table_version` parameter recorded on each run.

## C8. Optional — a serving endpoint

**Serving** → **Create serving endpoint** → select `<catalog>.bl.bl_brand_ranker` and the
version carrying `@champion`.

The container must install torch, CatBoost, TabPFN and `names-dataset`, then download
the TabPFN weights and fit the cached context — roughly **3 GB of memory and a two-minute
cold start**. On Free Edition's CPU-only serving this may be slow to start or may not fit.
Part A covers the serving path; what Databricks adds is the scheduled pipeline, Unity
Catalog registration and the job history.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `WinError 10013` on start | Port in use. Use `--port 8088` (that is why the default demo uses it). |
| Endpoint returns 503 | Still loading. Cold start is ~130 s; wait for `Application startup complete`. |
| Training looks frozen | It is not. CatBoost then TabPFN, ~8 min for `train_test`. Progress goes to the researcher's log file, not stdout. |
| `SyntaxError` in `bl_models_train.py` | You are on Python 3.11 or older. The researcher's script needs **3.12+** (PEP 701 f-strings). Use `.venv`, which is 3.12.10. |
| MLflow shows no runs | Check `MLFLOW_TRACKING_URI` is set in the terminal you are training from. Without it, runs go to a local `mlruns/` folder instead of the server. |
| `bundle validate` complains about clusters | Confirm you are on this version of `databricks/databricks.yml` — it must contain no `job_clusters` and no `node_type_id`. |
| Databricks job fails importing `tabpfn` | Serverless dependency resolution. Check the `environments.spec.dependencies` list in the bundle and the job's run log. |
| `DELTA_TABLE_URI is set ... but no SparkSession` | You set a UC table name while running off-cluster. Unset it for local runs. |
