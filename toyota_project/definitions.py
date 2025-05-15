from dagster import (
    Definitions,
    ScheduleDefinition,
    define_asset_job,
    load_assets_from_modules,
)

from toyota_project import assets  # noqa: TID252

all_assets = load_assets_from_modules([assets])

all_sync_jobs = define_asset_job(
    name="all_sync_jobs",
    selection=all_assets,
)

every_day = ScheduleDefinition(
    job=all_sync_jobs,
    cron_schedule="0 0 * * *",  # Every day at midnight
)

defs = Definitions(
    assets=all_assets,
    jobs=[all_sync_jobs],
    schedules=[every_day],
)