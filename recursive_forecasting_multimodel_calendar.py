"""
Recursive forecasting multi-transformer
+ métricas completas
+ previsão
+ split train/test
+ decomposição sazonal (EDA)
+ exógenas calendário
+ feature importance

python recursive_forecasting_multimodel.py
"""

# ======================================================
# IMPORTS
# ======================================================

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # evita erro tkinter
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.preprocessing import RollingFeatures

from statsmodels.tsa.seasonal import seasonal_decompose
import holidays


# ======================================================
# CONFIG
# ======================================================

DATA_PATH = "bases_tratadas/daily_peak_transformers_dataset.csv"

MAPA_TRANSFORMADORES = {
    "APL_DJ_12B1": "T3",
    "BSA_DJ_12B1": "T11a",
    "BSA_DJ_12B2": "T11b",
    "CBD_DJ_12B1": "T16a",
    "CBD_DJ_12B2": "T16b",
    "CPX_DJ_12B1": "T21a",
    "CPX_DJ_12B2": "T21b",
    "CRI_DJ_12B1": "T22a",
    "CRI_DJ_12B2": "T22b",
    "ILB_DJ_12B1": "T32",
    "JPS_DJ_12B1": "T36a",
    "JPS_DJ_12B2": "T36b",
    "MGB_DJ_12B1": "T42a",
    "MGB_DJ_12B2": "T42b",
    "TBU_DJ_12B1": "T70a",
    "TBU_DJ_12B2": "T70b",
}

TRANSFORMADORES_ALVO = list(MAPA_TRANSFORMADORES.values())

START_DATE = "2021-01-01"
END_VALIDATION = "2023-12-31"

LAGS = 365
STEPS = 365

OUTPUT_DIR = "resultados"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ======================================================
# MÉTRICAS
# ======================================================

def compute_metrics(y_true, y_pred):

    # mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    return rmse, mape, r2


# ======================================================
# PREPARO SÉRIE + EXÓGENAS
# ======================================================

def preparar_serie_diaria(df):

    serie = df.copy()
    serie["Time"] = pd.to_datetime(serie["datahora"])
    serie = serie.set_index("Time").sort_index()

    # índice diário completo
    full_index = pd.date_range(serie.index.min(), serie.index.max(), freq="D")
    serie = serie.reindex(full_index)

    serie = serie.loc[START_DATE:]

    # =========================
    # TARGET
    # =========================
    serie["Smax"] = (
        serie["Smax"]
        .interpolate(method="time")
        .ffill()
        .bfill()
    )

    # =========================
    # EXÓGENAS CALENDÁRIO
    # =========================
    br_holidays = holidays.Brazil()

    serie["weekday"] = serie.index.weekday
    serie["is_weekend"] = (serie["weekday"] >= 5).astype(int)
    serie["month"] = serie.index.month
    serie["dayofyear"] = serie.index.dayofyear
    serie["is_holiday"] = serie.index.isin(br_holidays).astype(int)

    y = serie["Smax"]

    exog = serie[
        ["weekday", "is_weekend", "month", "dayofyear", "is_holiday"]
    ]

    return y, exog


# ======================================================
# MODELOS
# ======================================================

def criar_modelos():

    return {
        "LGBM": LGBMRegressor(random_state=100, n_jobs=-1, verbose=-1),

        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            random_state=100,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            n_jobs=-1,
            verbosity=0
        ),

        "GradientBoosting": GradientBoostingRegressor(random_state=100)
    }


# ======================================================
# PLOTS
# ======================================================

def plot_split_real(y, transformer_id):

    split = pd.to_datetime(END_VALIDATION)

    plt.figure(figsize=(13, 5))
    y.loc[:split].plot(label="Train")
    y.loc[split:].plot(label="Test")
    plt.axvline(split, linestyle="--")

    plt.legend()
    plt.title(f"{transformer_id} - Train/Test split")

    plt.savefig(f"{OUTPUT_DIR}/plot_split_{transformer_id}.png")
    plt.close()


def plot_decomposition(y, transformer_id):

    result = seasonal_decompose(y, model="additive", period=365)

    fig = result.plot()
    fig.set_size_inches(12, 8)

    plt.savefig(f"{OUTPUT_DIR}/plot_decomposition_{transformer_id}.png")
    plt.close()


def plot_forecast(y, pred, transformer_id, modelo):

    split = pd.to_datetime(END_VALIDATION)

    plt.figure(figsize=(13, 5))

    y.loc[:split].plot(label="Train")
    y.loc[split:].plot(label="Test")
    pred["pred"].plot(label="Forecast")

    plt.axvline(split, linestyle="--")

    plt.legend()
    plt.title(f"{transformer_id} - {modelo}")

    plt.savefig(f"{OUTPUT_DIR}/plot_forecast_{transformer_id}_{modelo}.png")
    plt.close()


def plot_importance(model, exog_cols, transformer_id, nome):

    if not hasattr(model, "feature_importances_"):
        return

    lag_cols = [f"lag_{i}" for i in range(1, LAGS + 1)]
    cols = lag_cols + list(exog_cols)

    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(cols)), imp[order])
    plt.xticks(range(len(cols)), np.array(cols)[order], rotation=90)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/importance_{transformer_id}_{nome}.png")
    plt.close()


# ======================================================
# TREINO
# ======================================================

def treinar_multimodelo(df_filtrado):

    modelos = criar_modelos()

    resultados = []

    for transformer_id, df_tr in df_filtrado.groupby("id"):

        print(f"\n>>> {transformer_id}")

        y, exog = preparar_serie_diaria(df_tr)

        plot_split_real(y, transformer_id)
        plot_decomposition(y, transformer_id)

        initial_train_size = (y.index <= END_VALIDATION).sum()

        cv = TimeSeriesFold(
            steps=STEPS,
            initial_train_size=initial_train_size,
            refit=False
        )

        for nome, est in modelos.items():

            print(f"   -> {nome}")

            forecaster = ForecasterRecursive(
                estimator=est,
                lags=LAGS,
                window_features=RollingFeatures(
                    stats=["mean"],
                    window_sizes=[365]
                )
            )

            metric, pred = backtesting_forecaster(
                forecaster=forecaster,
                y=y,
                exog=exog,
                cv=cv,
                metric="mean_absolute_error",
                verbose=False
            )

            # mae = float(metric.iloc[0]) if isinstance(metric, pd.Series) else float(metric)

            y_true = y.loc[pred.index]
            y_pred = pred["pred"]

            rmse, mape, r2 = compute_metrics(y_true, y_pred)

            resultados.append({
                "id": transformer_id,
                "modelo": nome,
                "RMSE": rmse,
                "MAPE": mape,
                "R2": r2
            })

            plot_forecast(y, pred, transformer_id, nome)
            plot_importance(est, exog.columns, transformer_id, nome)

    return pd.DataFrame(resultados)


# ======================================================
# MAIN
# ======================================================

def main():

    df = pd.read_csv(DATA_PATH, sep=";", encoding="latin-1")
    df = df[df["id"].isin(TRANSFORMADORES_ALVO)]

    resultados = treinar_multimodelo(df)

    resultados.to_csv(f"{OUTPUT_DIR}/metricas.csv", index=False)
    # print(resultados.sort_values(["id", "MAE"]))


if __name__ == "__main__":
    main()

