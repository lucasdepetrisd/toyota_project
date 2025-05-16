# toyota_project

### Como correr el proyecto
1. Clonar el repositorio:
    ```powershell
    git clone
    cd toyota_project
    ```

2. Crear el entorno virtual:
    ```powershell
    conda create -n toyota python=3.11
    conda activate toyota
    conda install dagster poetry
    ```
    Si conda no puede instalar proba agregando el canal conda-forge:
   ```powershell
   conda config --add channels conda-forge
   ```

4. Instalar dependencias:
    ```powershell
    pip install -e ".[dev]"
    ```

5. Iniciar el servidor:
    ```powershell
    dagster dev
    ```

6. Abrir el navegador en `http://localhost:3000` para ver el proyecto.

### Como replicar este proyecto

1. Setear el entorno virtual:
    ```powershell
    conda create -n toyota python=3.11
    conda activate toyota
    conda install dagster poetry
    ```

2. Crear scaffold de dagster:
    ```powershell
    dagster project scaffold --name toyota_project
    ```

3. Abrir IDE en la carpeta creada `toyota_project` o hacer `cd` a la carpeta.

4. Instalar dependencias e iniciar servidor:
    ```powershell
    pip install -e ".[dev]"
    poetry add pandas scikit-learn statsmodels mlflow
    dagster dev
    ```

5. Agregar a `definitions.py`:

    ```python
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
    ```

6. Agregar a `assets.py`:

    ```python
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
    ```

---

# toyota-project

This is a [Dagster](https://dagster.io/) project scaffolded with [`dagster project scaffold`](https://docs.dagster.io/guides/build/projects/creating-a-new-project).

## Getting started

First, install your Dagster code location as a Python package. By using the --editable flag, pip will install your Python package in ["editable mode"](https://pip.pypa.io/en/latest/topics/local-project-installs/#editable-installs) so that as you develop, local code changes will automatically apply.

```bash
pip install -e ".[dev]"
```

Then, start the Dagster UI web server:

```bash
dagster dev
```

Open http://localhost:3000 with your browser to see the project.

You can start writing assets in `toyota_project/assets.py`. The assets are automatically loaded into the Dagster code location as you define them.

## Development

### Adding new Python dependencies

You can specify new Python dependencies in `setup.py`.

### Unit testing

Tests are in the `toyota_project_tests` directory and you can run tests using `pytest`:

```bash
pytest toyota_project_tests
```

### Schedules and sensors

If you want to enable Dagster [Schedules](https://docs.dagster.io/guides/automate/schedules/) or [Sensors](https://docs.dagster.io/guides/automate/sensors/) for your jobs, the [Dagster Daemon](https://docs.dagster.io/guides/deploy/execution/dagster-daemon) process must be running. This is done automatically when you run `dagster dev`.

Once your Dagster Daemon is running, you can start turning on schedules and sensors for your jobs.

## Deploy on Dagster+

The easiest way to deploy your Dagster project is to use Dagster+.

Check out the [Dagster+ documentation](https://docs.dagster.io/dagster-plus/) to learn more.
