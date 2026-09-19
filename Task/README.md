# Task/

Place `bl_full_data.csv` here before running anything.

The dataset is ~50 MB and is distributed via the Drive link in the assignment briefing
rather than committed to this repository, so this directory is empty on a fresh clone.

Both the ingestion step and the load simulation read it from `Task/bl_full_data.csv`:

```bash
python -m bl_ranker.data.ingest
python -m loadtest.simulate --url http://localhost:8088/rank
```

See the [runbook](../docs/RUNBOOK.md) for the full sequence.
