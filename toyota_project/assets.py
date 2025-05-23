import pandas as pd
import numpy as np
from dagster import asset
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns


@asset
def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        "https://raw.githubusercontent.com/dodobeatle/dataeng-datos/refs/heads/main/ToyotaCorolla.csv",
        encoding="latin1"
    )
    return df


@asset
def eda(load_data: pd.DataFrame) -> pd.DataFrame:
    # Filtrar columnas relevantes
    columnas = ["Price", "Age_08_04", "KM", "cc", "Doors", "Weight",
                "Automatic", "Fuel_Type", "Met_Color", "Quarterly_Tax"]
    df_selected = load_data[columnas].copy()

    # Pairplot
    sns.pairplot(data=df_selected.select_dtypes(include=np.number))
    plt.suptitle("Pairplot - Features Seleccionadas", y=1.02)
    plt.tight_layout()
    plt.show()

    # Matriz de correlación
    corr_matrix = df_selected.select_dtypes(
        include=np.number).corr(method='pearson')
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matriz de Correlación - Features Seleccionadas')
    plt.tight_layout()
    plt.show()

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
    print("🔎 Outliers detectados:")
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
        plt.tight_layout()
    plt.suptitle('Boxplots - Features Seleccionadas', fontsize=16, y=1.02)
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    plt.tight_layout()
    plt.show()

    # Scatterplots
    plt.figure(figsize=(15, 10))
    for i, column in enumerate([c for c in columnas if c != "Price"]):
        plt.subplot(3, 3, i + 1)
        sns.scatterplot(x=df_selected[column], y=df_selected["Price"])
        plt.title(f'{column} vs Precio')
        plt.tight_layout()
    plt.suptitle('Scatterplots de Features vs Precio', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # ✅ Devolver el DataFrame filtrado para que lo use el siguiente asset si es necesario
    return df_selected


@asset
def preparar_datos(eda: pd.DataFrame) -> dict:
    df = eda.copy()

    # Codificación
    df["Automatic"] = df["Automatic"].map({"Yes": 1, "No": 0})
    df = pd.get_dummies(df, columns=["Fuel_Type"], drop_first=True)

    # Variables predictoras y target
    features = ['Age_08_04', 'KM', 'cc', 'Doors', 'Weight', 'Automatic',
                'Met_Color', 'Quarterly_Tax', 'Fuel_Type_Diesel', 'Fuel_Type_Petrol']

    X = df[features]
    y = df["Price"]

    # División train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }


@asset
def entrenar_modelo(preparar_datos: dict) -> dict:
    X_train = preparar_datos["X_train"]
    y_train = preparar_datos["y_train"]
    X_test = preparar_datos["X_test"]
    y_test = preparar_datos["y_test"]

    # 🔍 Eliminar columnas completamente vacías
    cols_before = set(X_train.columns)
    X_train = X_train.dropna(axis=1, how='all')
    X_test = X_test[X_train.columns]  # asegurar que tenga las mismas columnas

    # ⚠️ Imputar valores faltantes con la media
    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # 📌 Alinear y_train con X_train imputado
    y_train_aligned = y_train.loc[X_train_imputed.index]

    # 🔁 Entrenar el modelo
    modelo = LinearRegression()
    modelo.fit(X_train_imputed, y_train_aligned)

    return {
        "modelo": modelo,
        "X_test": X_test_imputed,
        "y_test": y_test
    }


@asset
def evaluar_modelo(entrenar_modelo: dict):
    modelo = entrenar_modelo["modelo"]
    X_test = entrenar_modelo["X_test"]
    y_test = entrenar_modelo["y_test"]

    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    rss = np.sum((y_test - y_pred) ** 2)

    print(f"📈 MSE: {mse:.2f}")
    print(f"📈 RMSE: {rmse:.2f}")
    print(f"📈 MAE: {mae:.2f}")
    print(f"📈 RSS: {rss:.2f}")


@asset
def analizar_residuales(entrenar_modelo: dict):
    modelo = entrenar_modelo["modelo"]
    X_test = entrenar_modelo["X_test"]
    y_test = entrenar_modelo["y_test"]

    y_pred = modelo.predict(X_test)
    residuals = y_test - y_pred

    plt.figure(figsize=(10, 5))
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Precio predicho")
    plt.ylabel("Residuales")
    plt.title("Análisis de residuales")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
