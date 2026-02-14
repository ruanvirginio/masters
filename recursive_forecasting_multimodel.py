"""Treinamento recursivo multi-transformador com LGBM, XGBoost e GradientBoosting.

Uso:
    python recursive_forecasting_multimodel.py
"""

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from xgboost import XGBRegressor


DATA_PATH = "bases_tratadas/daily_peak_transformers_dataset.csv"
TRANSFORMADORES_ALVO = ["T21a", "T22a", "T70"]
END_VALIDATION = "2023-12-31 23:59:00"


def preparar_serie_diaria(df_transformador: pd.DataFrame) -> pd.Series:
    serie = df_transformador.copy()
    serie["Time"] = pd.to_datetime(serie["datahora"])
    serie = serie.set_index("Time").sort_index()

    full_index = pd.date_range(start=serie.index.min(), end=serie.index.max(), freq="D")
    serie = serie.reindex(full_index)
    serie["Smax"] = serie["Smax"].interpolate(method="time").ffill().bfill()

    return serie["Smax"]


def treinar_multimodelo(df_filtrado: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    modelos = {
        "LGBM": LGBMRegressor(random_state=100, verbose=-1),
        "XGBoost": XGBRegressor(
            random_state=100,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            verbosity=0,
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=100),
    }

    resultados_metricas = []
    predicoes_modelos = {}

    for transformer_id, df_transformer in df_filtrado.groupby("id"):
        y = preparar_serie_diaria(df_transformer)

        cv = TimeSeriesFold(
            steps=365,
            initial_train_size=len(y.loc[:END_VALIDATION]),
            refit=False,
        )

        for nome_modelo, estimador in modelos.items():
            forecaster = ForecasterRecursive(
                estimator=estimador,
                lags=365,
                window_features=RollingFeatures(stats=["mean"], window_sizes=365),
            )

            metrica, pred = backtesting_forecaster(
                forecaster=forecaster,
                y=y,
                cv=cv,
                metric="mean_absolute_error",
                verbose=False,
            )

            mae = (
                float(metrica["mean_absolute_error"])
                if isinstance(metrica, pd.DataFrame)
                else float(metrica)
            )

            resultados_metricas.append(
                {
                    "id": transformer_id,
                    "modelo": nome_modelo,
                    "MAE": mae,
                }
            )
            predicoes_modelos[(transformer_id, nome_modelo)] = pred

    resultados = pd.DataFrame(resultados_metricas).sort_values(["id", "MAE"])
    return resultados, predicoes_modelos


def main() -> None:
    df_daily = pd.read_csv(DATA_PATH, sep=";", encoding="latin-1")
    df_filtrado = df_daily.loc[df_daily["id"].isin(TRANSFORMADORES_ALVO)].copy()

    resultados, _ = treinar_multimodelo(df_filtrado)
    print("\n=== Métricas (MAE) por transformador/modelo ===")
    print(resultados.to_string(index=False))


if __name__ == "__main__":
    main()
