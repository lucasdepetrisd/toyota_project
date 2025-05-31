import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
from dagster import asset
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, Lasso, Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri("http://localhost:5000")

@asset(group_name="inicio")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        "https://raw.githubusercontent.com/dodobeatle/dataeng-datos/refs/heads/main/ToyotaCorolla.csv",
        encoding="latin1"
    )
    return df


@asset(group_name="inicio")
def limpiar_datos(load_data: pd.DataFrame) -> pd.DataFrame:
    df = load_data.copy()

    # Diagnóstico opcional (imprimí si lo querés ver en Dagster logs)
    print("Nulos por columna:")
    print(df.isnull().sum())

    print("Filas con valores negativos en columnas clave:")
    print(df[(df["KM"] < 0) | (df["Weight"] < 0) | (
        df["Age_08_04"] < 0) | (df["Price"] < 0)])

    # Elimino duplicados (excepto por 'Id' y 'Model')
    df = df.drop_duplicates(
        subset=[col for col in df.columns if col not in ["Id", "Model"]], keep='first')

    # Limpieza de texto en 'Model'
    df['Model'] = df['Model'].str.upper().str.lstrip('?')

    # Codificación de 'Fuel_Type' a dummies no booleanas
    df = pd.get_dummies(df, columns=["Fuel_Type"], drop_first=True, dtype=int)

    # Feature engineering: edad del auto en meses (al mes 08/2004)
    df["Age_08_04_calculado"] = (
        2004 - df["Mfg_Year"]) * 12 + (8 - df["Mfg_Month"])

    # Drop columnas no necesarias
    df.drop(columns=["Mfg_Year", "Mfg_Month", "Age_08_04", "Id"], inplace=True)

    # Combinar columnas de airbags
    df['Tiene_Airbag'] = ((df['Airbag_1'] == 1) | (
        df['Airbag_2'] == 1)).astype(int)
    df.drop(columns=["Airbag_1", "Airbag_2"], inplace=True)

    return df


@asset(group_name="inicio")
def eda(limpiar_datos: pd.DataFrame) -> pd.DataFrame:
    columnas = ["Price", "Age_08_04_calculado", "HP", "cc", "Doors", "Mfr_Guarantee",
                "Guarantee_Period", "ABS", "Airco", "Automatic_airco", "Boardcomputer",
                "Powered_Windows"]
    df_selected = limpiar_datos[columnas].copy()

    # Pairplot
    plt.figure()
    sns.pairplot(data=df_selected.select_dtypes(include=np.number))
    plt.suptitle("Pairplot - Todas las Features Numéricas", y=1.02)
    plt.tight_layout()
    # plt.show()

    # Matriz de correlación
    corr_matrix = df_selected.select_dtypes(
        include=np.number).corr(method='pearson')
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matriz de Correlación - Todas las Features Numéricas')
    plt.tight_layout()
    # plt.show()

    # Detección de outliers con IQR
    def detect_outliers(df_num):
        outliers = pd.DataFrame(columns=['Feature', 'Number of Outliers'])
        for column in df_num.columns:
            q1 = df_num[column].quantile(0.25)
            q3 = df_num[column].quantile(0.75)
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            n_outliers = df_num[(df_num[column] < low) |
                                (df_num[column] > high)].shape[0]
            outliers = pd.concat(
                [outliers, pd.DataFrame(
                    {'Feature': [column], 'Number of Outliers': [n_outliers]})],
                ignore_index=True
            )
        return outliers

    outliers_df = detect_outliers(df_selected.select_dtypes(include=np.number))
    print("Outliers detectados:")
    print(outliers_df)

    # Boxplots
    num_cols = df_selected.select_dtypes(include=np.number).columns.tolist()
    n = len(num_cols)
    cols = 3
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, column in enumerate(num_cols):
        plt.subplot(rows, cols, i + 1)
        sns.boxplot(y=df_selected[column])
        plt.title(f'Boxplot de {column}')
    plt.suptitle('Boxplots - Todas las Features Numéricas',
                 fontsize=16, y=1.02)
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    plt.tight_layout()
    plt.savefig("boxplots_all_features.png")
    # plt.show()

    # Scatterplots contra Price (si existe)
    if "Price" in df_selected.columns:
        plt.figure(figsize=(15, 10))
        cols_vs_price = [c for c in num_cols if c != "Price"]
        for i, column in enumerate(cols_vs_price[:9]):  # Máx. 9 gráficos
            plt.subplot(3, 3, i + 1)
            sns.scatterplot(x=df_selected[column], y=df_selected["Price"])
            plt.title(f'{column} vs Precio')
        plt.suptitle('Scatterplots de Features Numéricas vs Precio',
                     fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig("scatterplots_vs_price.png")
        # plt.show()
    
    plt.close()

    return df_selected


@asset(group_name="preparacion_datos")
def preparar_datos_ols(limpiar_datos: pd.DataFrame) -> dict:
    columnas = ["Price", "Age_08_04_calculado", "HP", "cc", "Doors", "Mfr_Guarantee",
                "Guarantee_Period", "ABS", "Airco", "Automatic_airco", "Boardcomputer",
                "Powered_Windows"]
    df = limpiar_datos[columnas].copy()

    # Imputación de valores nulos con la mediana
    imputer = SimpleImputer(strategy="median")
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    df_limpio = df[
        (df["Price"] > 1000) & (df["Price"] < 25000) &
        (df["Age_08_04_calculado"] > 6) & (df["Age_08_04_calculado"] < 80) &
        (df["cc"] > 600) & (df["cc"] < 2200) &
        (df["Guarantee_Period"] > 2) & (df["Guarantee_Period"] < 12) &
        (df["Doors"].isin([3, 4, 5])) &
        (df["HP"] > 30) & (df["HP"] < 118)
    ]

    features = [col for col in df_limpio.columns if col != "Price"]
    X = df_limpio[features]
    y = df_limpio["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=1
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "features": features
    }


@asset(group_name="preparacion_datos")
def preparar_datos_general(limpiar_datos: pd.DataFrame) -> dict:
    df = limpiar_datos.copy()

    df.drop(columns=["Model"], inplace=True)

    # Imputación de valores nulos con la mediana
    imputer = SimpleImputer(strategy="median")
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    features = [col for col in df_imputed.columns if col != "Price"]
    X = df_imputed[features]
    y = df_imputed["Price"]

    df_clean = df_imputed.copy()
    features_numericas = df_clean.select_dtypes(
        include='number').columns.drop("Price")
    for col in features_numericas:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower) &
                            (df_clean[col] <= upper)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
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
def entrenar_ols(preparar_datos_ols: dict) -> dict:
    model = LinearRegression()
    model.fit(preparar_datos_ols["X_train"], preparar_datos_ols["y_train"])
    return {"modelo": model, **preparar_datos_ols}


@asset(group_name="ols")
def evaluar_ols(entrenar_ols: dict) -> dict:
    with mlflow.start_run(experiment_id=mlflow.set_experiment("OLS").experiment_id, run_name="ols_run"):
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
def entrenar_lasso(preparar_datos_general: dict) -> dict:
    model = LassoCV(cv=5, random_state=42)
    model.fit(preparar_datos_general["X_train"],
              preparar_datos_general["y_train"])
    return {"modelo": model, **preparar_datos_general}


@asset(group_name="lasso")
def evaluar_lasso(entrenar_lasso: dict) -> dict:
    with mlflow.start_run(experiment_id=mlflow.set_experiment("Lasso").experiment_id, run_name="lasso_run"):
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


@asset(group_name="lasso")
def graficar_lasso(entrenar_lasso: dict) -> None:
    # Recupero datos
    X_train_scaled = entrenar_lasso["X_train"]
    y_train = entrenar_lasso["y_train"]
    features = entrenar_lasso["features"]
    feature_names = features  # lista de nombres de columnas

    # Trayectoria de coeficientes
    alphas = np.logspace(-1, 4, 100)
    coefs = []

    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_train_scaled, y_train)
        coefs.append(lasso.coef_)

    coefs = np.array(coefs)

    # Gráfico trayectoria coeficientes
    plt.figure(figsize=(12, 6))
    for i in range(len(feature_names)):
        plt.plot(alphas, coefs[:, i], label=feature_names[i])

    plt.xscale('log')
    plt.xlabel("Lambda (α)")
    plt.ylabel("Coeficientes Estandarizados")
    plt.title("Trayectorias de Coeficientes - Lasso")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("alphas_lasso.png")
    # plt.show()

    # Evaluación con modelo entrenado
    modelo = entrenar_lasso["modelo"]
    X_test_scaled = entrenar_lasso["X_test"]
    y_test = entrenar_lasso["y_test"]

    y_train_pred = modelo.predict(X_train_scaled)
    y_test_pred = modelo.predict(X_test_scaled)

    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    coeficientes = pd.Series(modelo.coef_, index=feature_names)

    # Reporte texto
    report = f"""
Alpha óptimo encontrado: {modelo.alpha_}

--- Entrenamiento ---
MSE: {mse_train:.4f}
R²: {r2_train:.4f}

--- Test ---
MSE: {mse_test:.4f}
R²: {r2_test:.4f}

Coeficientes seleccionados (!= 0):
{coeficientes[coeficientes != 0].to_string()}
"""

    filename = "lasso_report.txt"
    with open(filename, "w") as f:
        f.write(report)

    # Gráfico barras coeficientes ordenados por valor absoluto
    coef_sort = coeficientes.reindex(coeficientes.abs().sort_values(ascending=False).index)

    plt.figure(figsize=(10, 6))
    coef_sort.plot(kind='bar', color='salmon', edgecolor='black')
    plt.title("Importancia de Features según Lasso Regression")
    plt.ylabel("Valor del Coeficiente")
    plt.xlabel("Features")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("coeficientes_lasso.png")
    # plt.show()

    # Loggeo a MLflow
    with mlflow.start_run(run_name="lasso_graficar_run"):
        mlflow.log_artifact(filename)
        mlflow.log_artifact("coeficientes_lasso.png")
        mlflow.log_param("alpha_optimo", modelo.alpha_)
        mlflow.log_metric("mse_train", mse_train)
        mlflow.log_metric("r2_train", r2_train)
        mlflow.log_metric("mse_test", mse_test)
        mlflow.log_metric("r2_test", r2_test)
    
    plt.close()


# --- Ridge ---
@asset(group_name="ridge")
def entrenar_ridge(preparar_datos_general: dict) -> dict:
    alphas = np.logspace(-4, 4, 100)
    model = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
    model.fit(preparar_datos_general["X_train"],
              preparar_datos_general["y_train"])
    return {"modelo": model, **preparar_datos_general}


@asset(group_name="ridge")
def evaluar_ridge(entrenar_ridge: dict) -> dict:
    with mlflow.start_run(experiment_id=mlflow.set_experiment("Ridge").experiment_id, run_name="ridge_run"):
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


@asset(group_name="ridge")
def graficar_ridge(entrenar_ridge: dict) -> None:
    modelo = entrenar_ridge["modelo"]
    X_train_scaled = entrenar_ridge["X_train"]
    y_train = entrenar_ridge["y_train"]
    X_test_scaled = entrenar_ridge["X_test"]
    y_test = entrenar_ridge["y_test"]
    feature_names = entrenar_ridge["features"]

    # 1. Gráfico trayectorias coeficientes para varios alphas
    alphas = np.logspace(-3, 5, 100)
    coefs = []
    X_scaled_full = np.vstack([X_train_scaled, X_test_scaled])
    y_full = pd.concat([y_train, y_test], axis=0).values

    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_scaled_full, y_full)
        coefs.append(ridge.coef_)

    coefs = np.array(coefs)

    plt.figure(figsize=(12, 6))
    for i in range(len(feature_names)):
        plt.plot(alphas, coefs[:, i], label=feature_names[i])
    plt.xscale('log')
    plt.xlabel("Lambda")
    plt.ylabel("Coeficientes Estandarizados")
    plt.title("Trayectorias de Coeficientes - Ridge")
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lambdas_ridge.png")
    # plt.show()

    # 2. Evaluación del modelo entrenado
    y_train_pred = modelo.predict(X_train_scaled)
    y_test_pred = modelo.predict(X_test_scaled)

    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    coeficientes = pd.Series(modelo.coef_, index=feature_names)

    # 3. Guardar reporte txt con coeficientes ordenados
    coef_sort = coeficientes.reindex(coeficientes.abs().sort_values(ascending=False).index)

    report = f"""
Alpha óptimo encontrado: {modelo.alpha_}

--- Entrenamiento ---
MSE: {mse_train:.4f}
R²: {r2_train:.4f}

--- Test ---
MSE: {mse_test:.4f}
R²: {r2_test:.4f}

Coeficientes ordenados por importancia:
{coef_sort.to_string()}
"""
    filename = "ridge_report.txt"
    with open(filename, "w") as f:
        f.write(report)

    # 4. Gráfico barras coeficientes ordenados
    plt.figure(figsize=(10, 6))
    coef_sort.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Importancia de Features según Ridge Regression")
    plt.ylabel("Valor del Coeficiente")
    plt.xlabel("Features")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("coeficientes_ridge.png")
    # plt.show()

    # 5. Loguear artefactos y métricas a MLflow
    with mlflow.start_run(run_name="ridge_graficar_run"):
        mlflow.log_artifact(filename)
        mlflow.log_artifact("coeficientes_ridge.png")
        mlflow.log_param("alpha_optimo", modelo.alpha_)
        mlflow.log_metric("mse_train", mse_train)
        mlflow.log_metric("r2_train", r2_train)
        mlflow.log_metric("mse_test", mse_test)
        mlflow.log_metric("r2_test", r2_test)
    
    plt.close()


# --- PCA + OLS ---
@asset(group_name="pca")
def entrenar_pca(preparar_datos_general: dict) -> dict:
    pca = PCA()
    X_train_pca = pca.fit_transform(preparar_datos_general["X_train"])
    X_test_pca = pca.transform(preparar_datos_general["X_test"])
    model = LinearRegression()
    model.fit(X_train_pca, preparar_datos_general["y_train"])

    # Gráfico PC1 vs PC2
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
                          c=preparar_datos_general["y_train"], cmap="viridis", s=10)
    plt.colorbar(scatter, label="y_train")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.title("Distribución en PC1 vs PC2")
    plt.tight_layout()
    plt.savefig("pca_pc1_pc2.png")
    plt.close()

    return {
        "modelo": model,
        "X_test": X_test_pca,
        "y_test": preparar_datos_general["y_test"],
        "explained_variance": pca.explained_variance_ratio_
    }


@asset(group_name="pca")
def evaluar_pca(entrenar_pca: dict) -> dict:
    with mlflow.start_run(experiment_id=mlflow.set_experiment("PCA+OLS").experiment_id, run_name="pca_ols_run"):
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
        mlflow.log_artifact("pca_pc1_pc2.png")

        print(f"PCA + OLS - MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
        print(f"Explained variance per PC: {var_exp}")

        return {"mse": mse, "mae": mae, "r2": r2, "explained_variance": var_exp, "modelo": modelo}


@asset(group_name="pca")
def graficar_varianza_pca(entrenar_pca: dict) -> None:
    explained_var_ratio = entrenar_pca["explained_variance"]
    cumulative_var_ratio = np.cumsum(explained_var_ratio)

    # Gráfico de varianza explicada acumulada
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_var_ratio) + 1), cumulative_var_ratio, marker='o', label='Varianza Acumulada')
    plt.xlabel("Número de Componentes")
    plt.ylabel("Varianza Explicada Acumulada")
    plt.title("Curva de Varianza Explicada (PCA)")
    plt.grid(True)

    for threshold in [0.80, 0.90, 0.95, 0.99]:
        plt.axhline(y=threshold, color='r', linestyle='--', linewidth=1)
        plt.text(1, threshold + 0.01, f"{int(threshold*100)}%", color='r')

    plt.tight_layout()
    plt.savefig("varianza_explicada_acumulada.png")
    plt.show()

    # Tabla de varianza explicada y acumulada
    varianza_por_componente = pd.DataFrame({
        "Componente": np.arange(1, len(explained_var_ratio) + 1),
        "Varianza Explicada": explained_var_ratio,
        "Varianza Acumulada": cumulative_var_ratio
    })

    print(varianza_por_componente)

    # Guardar tabla a txt
    report_txt = "varianza_explicada_pca.txt"
    with open(report_txt, "w") as f:
        f.write(varianza_por_componente.to_string(index=False))

    # Loguear artefactos a MLflow
    with mlflow.start_run(run_name="pca_varianza_grafico"):
        mlflow.log_artifact("varianza_explicada_acumulada.png")
        mlflow.log_artifact(report_txt)
        mlflow.log_param("varianza_explicada", explained_var_ratio.tolist())
        mlflow.log_param("varianza_acumulada", cumulative_var_ratio.tolist())

    plt.close()


# -- Comparar modelos PCA + OLS vs OLS --
@asset(group_name="comparacion")
def comparar_pca_ols(evaluar_pca: dict, evaluar_ols: dict) -> dict:
    print("Evaluar PCA keys:", evaluar_pca.keys())
    print("Evaluar PCA values:", evaluar_pca)

    print("Evaluar OLS keys:", evaluar_ols.keys())
    print("Evaluar OLS values:", evaluar_ols)

    pca_metrics = {
        "MSE": evaluar_pca.get("mse", None),
        "MAE": evaluar_pca.get("mae", None),
        "R2": evaluar_pca.get("r2", None)
    }
    ols_metrics = {
        "MSE": evaluar_ols.get("mse", None),
        "MAE": evaluar_ols.get("mae", None),
        "R2": evaluar_ols.get("r2", None)
    }

    for metric in ["MSE", "MAE", "R2"]:
        if pca_metrics[metric] is None or ols_metrics[metric] is None:
            print(f"Warning: {metric} missing in PCA or OLS metrics")
        else:
            print(f"{metric}: OLS = {ols_metrics[metric]:.4f}, PCA+OLS = {pca_metrics[metric]:.4f}")

    # Crear gráficos individuales para cada métrica
    if None not in pca_metrics.values() and None not in ols_metrics.values():
        labels = ["OLS", "PCA + OLS"]
        width = 0.4
        x = range(len(labels))

        for metric in ["MSE", "MAE", "R2"]:
            ols_val = ols_metrics[metric]
            pca_val = pca_metrics[metric]

            plt.figure(figsize=(6, 4))
            plt.bar(x, [ols_val, pca_val], width, color=['blue', 'orange'])
            plt.xticks(x, labels)
            plt.ylabel(metric)
            plt.title(f"Comparación {metric}: OLS vs PCA + OLS")
            plt.tight_layout()
            filename = f"comparacion_{metric.lower()}.png"
            plt.savefig(filename)
            plt.show()
            plt.close()

            mlflow.log_artifact(filename)

        mlflow.log_metrics({
            "comparacion_mse_diff": pca_metrics["MSE"] - ols_metrics["MSE"],
            "comparacion_mae_diff": pca_metrics["MAE"] - ols_metrics["MAE"],
            "comparacion_r2_diff": pca_metrics["R2"] - ols_metrics["R2"],
        })
    
    plt.close()

    return {"pca_metrics": pca_metrics, "ols_metrics": ols_metrics}


# --- Comparar todos ---
@asset(group_name="comparacion")
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
        "PCA+OLS": evaluar_pca["mse"]
    }

    mejor_modelo = min(resultados, key=resultados.get)
    print(f"Mejor modelo según MSE: {mejor_modelo} con MSE={resultados[mejor_modelo]:.4f}")
    return mejor_modelo


# --- Analisis de Residuales ---
@asset(group_name="comparacion")
def analizar_residuales(entrenar_ols: dict) -> None:
    modelo = entrenar_ols["modelo"]
    X_test = entrenar_ols["X_test"]
    y_test = entrenar_ols["y_test"]

    y_pred = modelo.predict(X_test)
    residuales = y_test - y_pred

    # Gráfico de residuales
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuales)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.xlabel("Predicciones")
    plt.ylabel("Residuales")
    plt.title("Gráfico de Residuales vs Predicciones")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("residuales_vs_predicciones.png")
    plt.show()

    # Histograma de residuales
    plt.figure(figsize=(10, 6))
    sns.histplot(residuales, bins=30, kde=True)
    plt.xlabel("Residuales")
    plt.title("Histograma de Residuales")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("histograma_residuales.png")
    plt.show()

    # Loguear artefactos a MLflow
    with mlflow.start_run(run_name="analisis_residuales_run"):
        mlflow.log_artifact("residuales_vs_predicciones.png")
        mlflow.log_artifact("histograma_residuales.png")

    plt.close()