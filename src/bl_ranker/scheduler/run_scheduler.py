"""Weekly production training, Sunday 05:00.

The briefing requires the production run on a weekly Sunday 05:00 schedule. On Databricks
that is a Jobs schedule (see `databricks/databricks.yml`); locally it is this process.

APScheduler rather than Airflow: the workload is one task, once a week. Airflow would add
a scheduler, a webserver, a metadata database and roughly a gigabyte of containers to
express a single cron expression. The cron string here is identical to the one in the
Databricks job, so the two environments cannot silently diverge.

`misfire_grace_time` matters: if the host is asleep at 05:00 on Sunday -- entirely normal
for a laptop, and possible for a worker node -- the run fires late rather than being
skipped silently.
"""
from __future__ import annotations

import logging
import sys

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from bl_ranker.config import get_settings
from bl_ranker.training.run import run

log = logging.getLogger(__name__)


def weekly_production_run() -> None:
    log.info("Scheduled production training run starting")
    try:
        run_id = run(mode="production")
        log.info("Scheduled run finished: %s", run_id)
    except Exception:
        # Never let a failed run kill the scheduler; the next week must still fire.
        log.exception("Scheduled production run FAILED")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    settings = get_settings()
    trigger = CronTrigger.from_crontab(settings.schedule_cron, timezone=settings.schedule_timezone)

    scheduler = BlockingScheduler(timezone=settings.schedule_timezone)
    scheduler.add_job(
        weekly_production_run,
        trigger=trigger,
        id="bl_weekly_production_training",
        max_instances=1,
        coalesce=True,          # if several fires are pending, run once
        misfire_grace_time=6 * 3600,
    )
    log.info(
        "Scheduler started: cron='%s' tz=%s. Next run: %s",
        settings.schedule_cron,
        settings.schedule_timezone,
        trigger.get_next_fire_time(None, __import__("datetime").datetime.now()),
    )
    scheduler.start()


if __name__ == "__main__":
    main()
