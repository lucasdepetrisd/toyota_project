from dagster import Definitions
from .assets import (
    load_data,
    eda,
    preparar_datos,
    entrenar_modelo,
    evaluar_modelo,
    analizar_residuales,
)

defs = Definitions(
    assets=[
        load_data,
        eda,
        preparar_datos,
        entrenar_modelo,
        evaluar_modelo,
        analizar_residuales,
    ]
)


