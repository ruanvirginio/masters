# =============================================================================
# stats_tests.py — Testes Estatísticos (correções #13, #14, #17)
# =============================================================================
# Orquestrador de:
#   #13 — Diagnósticos formais do SARIMAX (Ljung-Box, ADF, KPSS, Jarque-Bera)
#   #14 — Block bootstrap 95% CI das métricas de teste
#   #17 — Diebold-Mariano par-a-par entre todos os modelos
#
# Pré-requisitos: as previsões de teste de cada modelo precisam estar salvas
# como CSV no formato: data,y_true,y_pred — em <OUTPUT_*>/preds_test_<tid>_<model>.csv
#
# Cada um dos models_*.py salva esses CSVs (foi adicionada a chamada
# salvar_preds_teste() no fim do loop de cada transformador).
# =============================================================================
import warnings; warnings.filterwarnings("ignore")
import os, glob
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

from config import (
    DATA_PATH, TRANSFORMADORES, MAPA_PLOT,
    OUTPUT_SARIMAX, OUTPUT_UV, OUTPUT_MV, OUTPUT_DL, OUTPUT_SVR,
    OUTPUT_REPORT,
    BOOTSTRAP_BLOCK_SIZE, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_ALPHA
)
from utils import (
    preparar_serie_diaria, get_splits,
    block_bootstrap_metric_ci, diebold_mariano_test,
    log
)

OUTPUT_STATS = "resultados_stats"
os.makedirs(OUTPUT_STATS, exist_ok=True)

PRED_DIRS = [OUTPUT_SARIMAX, OUTPUT_UV, OUTPUT_MV, OUTPUT_DL, OUTPUT_SVR,
             "resultados_NAIVE"]


# =============================================================================
# UTIL: leitura das previsões salvas pelos models_*.py
# =============================================================================
def carregar_predicoes_teste() -> pd.DataFrame:
    """
    Concatena todos os CSVs preds_test_*.csv encontrados nos diretórios de
    resultado em um único DataFrame longo:
        date, y_true, y_pred, transformador, modelo
    """
    rows = []
    for d in PRED_DIRS:
        if not os.path.isdir(d):
            continue
        for f in glob.glob(os.path.join(d, "preds_test_*.csv")):
            base = os.path.basename(f).replace("preds_test_", "").replace(".csv", "")
            # Esperado: <tid>_<modelo>
            try:
                tid, *modelo_parts = base.split("_", 1)
                modelo = modelo_parts[0] if modelo_parts else "?"
            except Exception:
                continue
            try:
                df = pd.read_csv(f)
                df["transformador"] = tid
                df["modelo"]        = modelo
                rows.append(df)
            except Exception as e:
                log.warning(f"Falha ao ler {f}: {e}")
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


# =============================================================================
# CORREÇÃO #13 — Diagnósticos do SARIMAX
# =============================================================================
def diagnosticos_sarimax():
    """
    Para cada transformador, lê os resíduos do SARIMAX e aplica:
      - Ljung-Box (autocorrelação residual em lags 7, 14 e 30)
      - ADF (raiz unitária em y_train)
      - KPSS (estacionariedade em y_train)
      - Jarque-Bera (normalidade dos resíduos)

    O CSV de resíduos espera-se em OUTPUT_SARIMAX/residuos_<tid>.csv com
    colunas date,residual. O models_sarimax.py foi atualizado para salvá-lo.
    """
    df_full = pd.read_csv(DATA_PATH, sep=";", encoding="latin-1")
    df_full = df_full[df_full["id"].isin(TRANSFORMADORES)].copy()

    rows = []
    for tid, df_tr in df_full.groupby("id"):
        y      = preparar_serie_diaria(df_tr)
        splits = get_splits(y)
        y_train = y.loc[splits["train"]]

        rec = {"transformador": tid, "nome_plot": MAPA_PLOT.get(tid, tid)}

        # ── ADF (raiz unitária) ───────────────────────────────────────────
        try:
            adf_stat, adf_p, *_ = adfuller(y_train.dropna(), autolag="AIC")
            rec["ADF_stat"]    = round(float(adf_stat), 4)
            rec["ADF_pvalue"]  = round(float(adf_p), 4)
            rec["ADF_concl"]   = "stationary" if adf_p < 0.05 else "non-stationary"
        except Exception as e:
            rec["ADF_stat"] = rec["ADF_pvalue"] = np.nan
            rec["ADF_concl"] = f"error: {e}"

        # ── KPSS (estacionariedade) ───────────────────────────────────────
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # InterpolationWarning é esperado
                kpss_stat, kpss_p, *_ = kpss(y_train.dropna(), regression="c", nlags="auto")
            rec["KPSS_stat"]   = round(float(kpss_stat), 4)
            # Quando p extrapola tabela: <0.01 ou >0.10 são os limites reportáveis
            rec["KPSS_pvalue"] = round(float(kpss_p), 4)
            rec["KPSS_concl"]  = "stationary" if kpss_p > 0.05 else "non-stationary"
        except Exception as e:
            rec["KPSS_stat"] = rec["KPSS_pvalue"] = np.nan
            rec["KPSS_concl"] = f"error: {e}"

        # ── Resíduos do SARIMAX (Ljung-Box + Jarque-Bera) ─────────────────
        resid_path = os.path.join(OUTPUT_SARIMAX, f"residuos_{tid}.csv")
        if os.path.exists(resid_path):
            try:
                resid = pd.read_csv(resid_path)
                r = resid["residual"].dropna().values

                lb = acorr_ljungbox(r, lags=[7, 14, 30], return_df=True)
                rec["LjungBox_lag7_p"]   = round(float(lb["lb_pvalue"].iloc[0]), 4)
                rec["LjungBox_lag14_p"]  = round(float(lb["lb_pvalue"].iloc[1]), 4)
                rec["LjungBox_lag30_p"]  = round(float(lb["lb_pvalue"].iloc[2]), 4)
                # Conclusão: H0 = sem autocorrelação. p > 0.05 → resíduos OK.
                rec["LjungBox_concl"] = ("residuals_OK" if lb["lb_pvalue"].min() > 0.05
                                          else "autocorrelation_remaining")

                jb_stat, jb_p = jarque_bera(r)
                rec["JarqueBera_stat"]   = round(float(jb_stat), 4)
                rec["JarqueBera_pvalue"] = round(float(jb_p), 4)
                rec["JarqueBera_concl"]  = ("normal" if jb_p > 0.05
                                             else "non-normal")
            except Exception as e:
                rec["LjungBox_concl"] = f"error: {e}"
        else:
            log.warning(f"  Resíduos não encontrados para {tid}: {resid_path}")
            rec["LjungBox_concl"] = "no_residuals_file"

        rows.append(rec)

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_STATS, "sarimax_diagnostics.csv")
    df.to_csv(out_path, index=False)
    log.info(f"\n[#13] Diagnósticos SARIMAX salvos em {out_path}")
    print(df.to_string(index=False))
    return df


# =============================================================================
# CORREÇÃO #14 — Block bootstrap 95% CI das métricas de teste
# =============================================================================
def bootstrap_ci_todos_modelos():
    """
    Para cada (transformador × modelo) com previsões salvas, aplica
    block bootstrap (n=BOOTSTRAP_N_RESAMPLES, blocos de BOOTSTRAP_BLOCK_SIZE
    dias) e computa IC 95% para RMSE, MAE e sMAPE.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    def _rmse(yt, yp): return float(np.sqrt(mean_squared_error(yt, yp)))
    def _mae(yt, yp):  return float(mean_absolute_error(yt, yp))
    def _smape(yt, yp):
        eps = 1e-8
        return float(np.mean(2 * np.abs(yt - yp) /
                              (np.abs(yt) + np.abs(yp) + eps)) * 100)

    preds = carregar_predicoes_teste()
    if preds.empty:
        log.error("Nenhuma previsão de teste encontrada. "
                  "Rode os modelos primeiro:")
        log.error("  python models_sarimax.py && python models_naive.py "
                  "&& python models_svr.py && python models_boosting.py "
                  "&& python models_dl.py")
        log.error("Ou rode tudo de uma vez: python run_all_corrections.py")
        return pd.DataFrame()

    rows = []
    for (tid, modelo), grp in preds.groupby(["transformador", "modelo"]):
        grp = grp.sort_values("date")
        y_true = grp["y_true"].values
        y_pred = grp["y_pred"].values

        rmse_ci  = block_bootstrap_metric_ci(
            y_true, y_pred, _rmse,
            block_size=BOOTSTRAP_BLOCK_SIZE,
            n_resamples=BOOTSTRAP_N_RESAMPLES,
            alpha=BOOTSTRAP_ALPHA
        )
        mae_ci   = block_bootstrap_metric_ci(
            y_true, y_pred, _mae,
            block_size=BOOTSTRAP_BLOCK_SIZE,
            n_resamples=BOOTSTRAP_N_RESAMPLES,
            alpha=BOOTSTRAP_ALPHA
        )
        smape_ci = block_bootstrap_metric_ci(
            y_true, y_pred, _smape,
            block_size=BOOTSTRAP_BLOCK_SIZE,
            n_resamples=BOOTSTRAP_N_RESAMPLES,
            alpha=BOOTSTRAP_ALPHA
        )

        rows.append({
            "transformador": tid,
            "nome_plot":     MAPA_PLOT.get(tid, tid),
            "modelo":        modelo,
            "n_test":        len(y_true),
            "RMSE":          round(rmse_ci["point"],  4),
            "RMSE_lower95":  round(rmse_ci["lower"],  4),
            "RMSE_upper95":  round(rmse_ci["upper"],  4),
            "RMSE_std":      round(rmse_ci["std"],    4),
            "MAE":           round(mae_ci["point"],   4),
            "MAE_lower95":   round(mae_ci["lower"],   4),
            "MAE_upper95":   round(mae_ci["upper"],   4),
            "sMAPE":         round(smape_ci["point"], 4),
            "sMAPE_lower95": round(smape_ci["lower"], 4),
            "sMAPE_upper95": round(smape_ci["upper"], 4),
        })

    df = pd.DataFrame(rows).sort_values(["transformador", "RMSE"])
    out_path = os.path.join(OUTPUT_STATS, "bootstrap_ci_test_metrics.csv")
    df.to_csv(out_path, index=False)
    log.info(f"\n[#14] Block bootstrap CI salvo em {out_path}")
    print(df[["transformador", "modelo", "RMSE", "RMSE_lower95",
              "RMSE_upper95", "sMAPE", "sMAPE_lower95", "sMAPE_upper95"]]
          .to_string(index=False))
    return df


# =============================================================================
# CORREÇÃO #17 — Diebold-Mariano par-a-par
# =============================================================================
def diebold_mariano_pares():
    """
    Para cada transformador, faz DM par-a-par entre todos os modelos com
    previsões salvas. Salva matriz triangular superior:
       columns: transformador, modelo_1, modelo_2, DM, p_value, melhor
    """
    preds = carregar_predicoes_teste()
    if preds.empty:
        log.error("Nenhuma previsão de teste encontrada para DM. "
                  "Rode os modelos primeiro (ver mensagem em #14).")
        return pd.DataFrame()

    rows = []
    for tid, grp in preds.groupby("transformador"):
        # Pivot: cada coluna = previsão de um modelo
        pivot = grp.pivot_table(index="date",
                                columns="modelo",
                                values="y_pred",
                                aggfunc="first")
        # y_true é único por data
        ytrue_series = (grp.drop_duplicates("date")
                            .set_index("date")["y_true"]
                            .reindex(pivot.index))
        pivot = pivot.dropna(how="any")
        ytrue_aligned = ytrue_series.loc[pivot.index]

        modelos = list(pivot.columns)
        for i in range(len(modelos)):
            for j in range(i + 1, len(modelos)):
                m1, m2 = modelos[i], modelos[j]
                res = diebold_mariano_test(
                    ytrue_aligned.values,
                    pivot[m1].values,
                    pivot[m2].values,
                    h=1, loss="se"
                )
                if np.isnan(res["DM"]):
                    melhor = "n/a"
                else:
                    if res["p_value"] < 0.05:
                        # DM > 0 → m2 é melhor (loss menor)
                        melhor = m2 if res["DM"] > 0 else m1
                    else:
                        melhor = "no_significant_diff"
                rows.append({
                    "transformador": tid,
                    "nome_plot":     MAPA_PLOT.get(tid, tid),
                    "modelo_1":      m1,
                    "modelo_2":      m2,
                    "DM_stat":       round(res["DM"], 4) if not np.isnan(res["DM"]) else np.nan,
                    "p_value":       round(res["p_value"], 4) if not np.isnan(res["p_value"]) else np.nan,
                    "n_test":        res["n"],
                    "melhor_p<0.05": melhor,
                })

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_STATS, "diebold_mariano_pairs.csv")
    df.to_csv(out_path, index=False)
    log.info(f"\n[#17] Diebold-Mariano salvo em {out_path}")
    if not df.empty:
        print(df.to_string(index=False))
    return df


# =============================================================================
# MAIN
# =============================================================================
def main():
    log.info("="*70)
    log.info(" Testes estatísticos — correções #13, #14, #17 ")
    log.info("="*70)

    log.info("\n>>> #13 — Diagnósticos SARIMAX (Ljung-Box, ADF, KPSS, JB)")
    diagnosticos_sarimax()

    log.info("\n>>> #14 — Block bootstrap 95% CI dos erros de teste")
    bootstrap_ci_todos_modelos()

    log.info("\n>>> #17 — Diebold-Mariano par-a-par")
    diebold_mariano_pares()


if __name__ == "__main__":
    main()
