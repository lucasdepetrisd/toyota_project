from dagster import AssetIn, AssetKey, asset
import pandas as pd


@asset
def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        "https://raw.githubusercontent.com/dodobeatle/dataeng-datos/refs/heads/main/ToyotaCorolla.csv",
        encoding="utf8",
        engine="python",
    )
    return df


@asset(ins={"mis_datos": AssetIn(key=AssetKey("load_data"))})
def transform_data(mis_datos: pd.DataFrame):
    mis_datos.describe()
    return mis_datos.head(10)
