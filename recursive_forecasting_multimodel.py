"""
Recursive forecasting multi-transformer
+ métricas
+ previsão
+ split train/test
+ decomposição sazonal (EDA)

python recursive_forecasting_multimodel.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.preprocessing import RollingFeatures

from statsmodels.tsa.seasonal import seasonal_decompose


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
# PREPARO SÉRIE
# ======================================================

def preparar_serie_diaria(df):

    serie = df.copy()
    serie["Time"] = pd.to_datetime(serie["datahora"])
    serie = serie.set_index("Time").sort_index()

    full_index = pd.date_range(serie.index.min(), serie.index.max(), freq="D")
    serie = serie.reindex(full_index)

    # sem leakage
    serie = serie.loc[START_DATE:]
    serie["Smax"] = serie["Smax"].interpolate(method="time").ffill()

    return serie["Smax"]


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
# PLOTS EDA
# ======================================================

def plot_split_real(y, transformer_id):

    split = pd.to_datetime(END_VALIDATION)

    plt.figure(figsize=(13, 5))
    y.loc[:split].plot(label="Train")
    y.loc[split:].plot(label="Test")
    plt.axvline(split, linestyle="--")

    plt.title(f"{transformer_id} - Train/Test split")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/plot_split_{transformer_id}.png", dpi=150)
    plt.close()


def plot_decomposition(y, transformer_id):

    result = seasonal_decompose(
        y,
        model="additive",
        period=365
    )

    fig = result.plot()
    fig.set_size_inches(12, 8)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot_decomposition_{transformer_id}.png", dpi=150)
    plt.close()


def plot_forecast(y, pred, transformer_id, modelo):

    split = pd.to_datetime(END_VALIDATION)

    plt.figure(figsize=(13, 5))

    y.loc[:split].plot(label="Train")
    y.loc[split:].plot(label="Test")
    pred["pred"].plot(label="Forecast")

    plt.axvline(split, linestyle="--")

    plt.title(f"{transformer_id} - {modelo}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/plot_forecast_{transformer_id}_{modelo}.png", dpi=150)
    plt.close()


# ======================================================
# TREINO
# ======================================================

def treinar_multimodelo(df_filtrado):

    modelos = criar_modelos()

    resultados = []
    predicoes = {}

    for transformer_id, df_tr in df_filtrado.groupby("id"):

        print(f"\n>>> {transformer_id}")

        y = preparar_serie_diaria(df_tr)

        # -------- EDA --------
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
                cv=cv,
                metric="mean_absolute_error",
                verbose=False
            )

            mae = float(metric if not isinstance(metric, pd.DataFrame)
                        else metric["mean_absolute_error"])

            resultados.append({
                "id": transformer_id,
                "modelo": nome,
                "MAE": mae
            })

            predicoes[(transformer_id, nome)] = pred

            plot_forecast(y, pred, transformer_id, nome)

    return pd.DataFrame(resultados), predicoes


# ======================================================
# MAIN
# ======================================================

def main():

    df = pd.read_csv(DATA_PATH, sep=";", encoding="latin-1")
    df = df[df["id"].isin(TRANSFORMADORES_ALVO)]

    resultados, _ = treinar_multimodelo(df)

    resultados.to_csv(f"{OUTPUT_DIR}/metricas.csv", index=False)
    print(resultados.sort_values(["id", "MAE"]))


if __name__ == "__main__":
    main()
