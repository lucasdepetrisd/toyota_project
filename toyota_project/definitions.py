from dagster import load_assets_from_modules
from toyota_project import assets

all_assets = load_assets_from_modules([assets])
