from dagster import Definitions
from .assets import (
    load_data,
    preparar_datos,
    entrenar_ols,
    evaluar_ols,
    entrenar_lasso,
    evaluar_lasso,
    entrenar_ridge,
    evaluar_ridge,
    entrenar_pca,
    evaluar_pca,
    comparar_modelos,
)

defs = Definitions(
    assets=[
        load_data,
        preparar_datos,
        # OLS
        entrenar_ols,
        evaluar_ols,
        # Lasso
        entrenar_lasso,
        evaluar_lasso,
        # Ridge
        entrenar_ridge,
        evaluar_ridge,
        # PCA
        entrenar_pca,
        evaluar_pca,
        # Comparación
        comparar_modelos,
    ]
)
