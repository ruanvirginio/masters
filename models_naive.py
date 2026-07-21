# =============================================================================
# models_naive.py — Seasonal Naive Baseline (correção #9 do orientador)
# =============================================================================
# Baseline mínimo: ŷ_t = y_{t-period}, com period=7 (ciclo semanal dominante)
# e period=365 (ciclo anual, opcional).
#
# Sem hiperparâmetros, sem treino, sem validação — usa apenas o passado
# observado. Serve como teto inferior interpretável: se um modelo perder
# para o seasonal naive, ele não está aprendendo nada útil.
#
# Saída: previsão de teste (2024) + métricas. Acompanha o mesmo formato
# dos demais scripts para integrar diretamente na Tabela 7.
# =============================================================================
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd

from config import (
    DATA_PATH, TRANSFORMADORES, MAPA_PLOT,
    OUTPUT_REPORT
)
from utils import (
    preparar_serie_diaria, compute_metrics, get_splits,
    plot_forecast_3way, plot_residuals,
    seasonal_naive_forecast,
    log
)

OUTPUT_NAIVE = "resultados_NAIVE"
os.makedirs(OUTPUT_NAIVE, exist_ok=True)


def treinar_naive(df: pd.DataFrame, period: int = 7) -> pd.DataFrame:
    """
    Aplica seasonal naive (ŷ_t = y_{t-period}) ao conjunto de teste.
    Usa treino+validação como histórico conhecido.
    """
    resultados = []

    for tid, df_tr in df.groupby("id"):
        log.info(f"\n{'='*60}\n  {tid}  [SeasonalNaive period={period}]\n{'='*60}")
        y      = preparar_serie_diaria(df_tr)
        splits = get_splits(y)

        y_train_val = y.loc[splits["train"].union(splits["val"])]
        y_test      = y.loc[splits["test"]]

        # ── Previsão recursiva sobre o teste ──────────────────────────────
        pred_test = seasonal_naive_forecast(y_train_val, splits["test"], period=period)

        # ── Previsão de validação (apenas para o gráfico 3-way) ───────────
        # Usa só o treino como histórico
        pred_val = seasonal_naive_forecast(
            y.loc[splits["train"]], splits["val"], period=period
        )

        metrics = compute_metrics(y_test, pred_test)
        log.info(f"  ✓ TESTE → RMSE={metrics['RMSE']:.4f} "
                 f"MAE={metrics['MAE']:.4f} sMAPE={metrics['sMAPE']:.2f}% "
                 f"R²={metrics['R2']:.4f}")

        nome_modelo = f"SeasonalNaive_{period}d"
        plot_forecast_3way(y, pred_val, pred_test, tid, nome_modelo, OUTPUT_NAIVE)
        plot_residuals(y_test, pred_test, tid, nome_modelo, OUTPUT_NAIVE)

        resultados.append({
            "transformador": tid,
            "nome_plot":     MAPA_PLOT.get(tid, tid),
            "modelo":        f"SeasonalNaive_{period}d",
            "abordagem":     "Baseline",
            "best_params":   f"period={period}",
            **{k: round(v, 4) for k, v in metrics.items()},
        })

    return pd.DataFrame(resultados)


def main():
    df = pd.read_csv(DATA_PATH, sep=";", encoding="latin-1")
    df = df[df["id"].isin(TRANSFORMADORES)].copy()

    # Roda dois períodos: semanal (dominante) e anual (referência sazonal)
    res_w = treinar_naive(df, period=7)
    res_w.to_csv(f"{OUTPUT_NAIVE}/metricas_SeasonalNaive_7d.csv", index=False)

    res_y = treinar_naive(df, period=365)
    res_y.to_csv(f"{OUTPUT_NAIVE}/metricas_SeasonalNaive_365d.csv", index=False)

    res = pd.concat([res_w, res_y], ignore_index=True)
    res.to_csv(f"{OUTPUT_NAIVE}/metricas_SeasonalNaive.csv", index=False)
    print(res[["transformador", "modelo", "RMSE", "MAE", "sMAPE", "R2"]]
          .sort_values(["transformador", "modelo"]).to_string(index=False))


if __name__ == "__main__":
    main()
