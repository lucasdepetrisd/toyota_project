import pandas as pd
import numpy as np
import mlflow
from dagster import asset
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@asset
def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        "https://raw.githubusercontent.com/dodobeatle/dataeng-datos/refs/heads/main/ToyotaCorolla.csv",
        encoding="latin1"
    )
    return df


@asset
def preparar_datos(load_data: pd.DataFrame) -> dict:
    columnas = ["Price", "Age_08_04", "KM", "cc", "Doors", "Weight",
                "Automatic", "Fuel_Type", "Met_Color", "Quarterly_Tax"]
    df = load_data[columnas].copy()

    df["Automatic"] = df["Automatic"].map({"Yes": 1, "No": 0})
    df = pd.get_dummies(df, columns=["Fuel_Type"], drop_first=True)

    imputer = SimpleImputer(strategy="median")
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    features = [col for col in df_imputed.columns if col != "Price"]
    X = df_imputed[features]
    y = df_imputed["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "features": features
    }


# --- OLS ---
@asset(group_name="ols")
def entrenar_ols(preparar_datos: dict) -> dict:
    model = LinearRegression()
    model.fit(preparar_datos["X_train"], preparar_datos["y_train"])
    return {"modelo": model, **preparar_datos}


@asset(group_name="ols")
def evaluar_ols(entrenar_ols: dict) -> dict:
    modelo = entrenar_ols["modelo"]
    X_test = entrenar_ols["X_test"]
    y_test = entrenar_ols["y_test"]

    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metrics({"mse": mse, "mae": mae, "r2": r2})

    print(f"OLS - MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

    return {"mse": mse, "mae": mae, "r2": r2, "modelo": modelo}


# --- Lasso ---
@asset(group_name="lasso")
def entrenar_lasso(preparar_datos: dict) -> dict:
    model = LassoCV(cv=5, random_state=42)
    model.fit(preparar_datos["X_train"], preparar_datos["y_train"])
    return {"modelo": model, **preparar_datos}


@asset(group_name="lasso")
def evaluar_lasso(entrenar_lasso: dict) -> dict:
    modelo = entrenar_lasso["modelo"]
    X_test = entrenar_lasso["X_test"]
    y_test = entrenar_lasso["y_test"]

    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metrics({"mse": mse, "mae": mae, "r2": r2})
    mlflow.log_param("alpha", modelo.alpha_)

    print(f"Lasso - MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}, Alpha: {modelo.alpha_:.5f}")

    return {"mse": mse, "mae": mae, "r2": r2, "alpha": modelo.alpha_, "modelo": modelo}


# --- Ridge ---
@asset(group_name="ridge")
def entrenar_ridge(preparar_datos: dict) -> dict:
    alphas = np.logspace(-4, 4, 100)
    model = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
    model.fit(preparar_datos["X_train"], preparar_datos["y_train"])
    return {"modelo": model, **preparar_datos}


@asset(group_name="ridge")
def evaluar_ridge(entrenar_ridge: dict) -> dict:
    modelo = entrenar_ridge["modelo"]
    X_test = entrenar_ridge["X_test"]
    y_test = entrenar_ridge["y_test"]

    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metrics({"mse": mse, "mae": mae, "r2": r2})
    mlflow.log_param("alpha", modelo.alpha_)

    print(f"Ridge - MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}, Alpha: {modelo.alpha_:.5f}")

    return {"mse": mse, "mae": mae, "r2": r2, "alpha": modelo.alpha_, "modelo": modelo}


# --- PCA + OLS ---
@asset(group_name="pca")
def entrenar_pca(preparar_datos: dict) -> dict:
    pca = PCA(n_components=5)
    X_train_pca = pca.fit_transform(preparar_datos["X_train"])
    X_test_pca = pca.transform(preparar_datos["X_test"])
    model = LinearRegression()
    model.fit(X_train_pca, preparar_datos["y_train"])
    return {
        "modelo": model,
        "X_test": X_test_pca,
        "y_test": preparar_datos["y_test"],
        "explained_variance": pca.explained_variance_ratio_
    }


@asset(group_name="pca")
def evaluar_pca(entrenar_pca: dict) -> dict:
    modelo = entrenar_pca["modelo"]
    X_test = entrenar_pca["X_test"]
    y_test = entrenar_pca["y_test"]
    var_exp = entrenar_pca["explained_variance"]

    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metrics({"mse": mse, "mae": mae, "r2": r2})
    mlflow.log_param("explained_variance", var_exp.tolist())

    print(f"PCA + OLS - MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
    print(f"Explained variance per PC: {var_exp}")

    return {"mse": mse, "mae": mae, "r2": r2, "explained_variance": var_exp, "modelo": modelo}


# --- Comparar todos ---
@asset
def comparar_modelos(
    evaluar_ols: dict,
    evaluar_lasso: dict,
    evaluar_ridge: dict,
    evaluar_pca: dict
) -> str:
    resultados = {
        "OLS": evaluar_ols["mse"],
        "Lasso": evaluar_lasso["mse"],
        "Ridge": evaluar_ridge["mse"],
        "PCA": evaluar_pca["mse"],
    }

    mejor_modelo = min(resultados, key=resultados.get)
    mejor_mse = resultados[mejor_modelo]

    print(f"\n🏆 Mejor modelo según MSE: {mejor_modelo} con MSE = {mejor_mse:.2f}")

    return mejor_modelo


