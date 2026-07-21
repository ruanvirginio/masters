# =============================================================================
# build_table7.py — Reformatar Tabela 7 (correção #15 do orientador)
# =============================================================================
# Lê:
#   - resultados_DL/runs_detail_<tid>_<arch>_<mode>.csv  (média ± dp dos N runs)
#   - resultados_*/preds_test_*.csv                     (previsões de teste)
#   - resultados_stats/diebold_mariano_pairs.csv        (p-values DM par-a-par)
#   - resultados_stats/bootstrap_ci_test_metrics.csv    (IC bootstrap)
#
# Gera:
#   - relatorio/table7_final.csv : tabela com (i) média±dp dos runs DL,
#                                  (ii) RMSE pontual + IC95 bootstrap dos demais,
#                                  (iii) p-value DM contra o melhor de cada linha,
#                                  (iv) marca em negrito todos os modelos
#                                       NÃO significativamente piores que o melhor.
#   - relatorio/table7_final.tex : versão LaTeX pronta para colar no paper.
# =============================================================================
import warnings; warnings.filterwarnings("ignore")
import os, glob
import numpy as np
import pandas as pd

from config import (
    TRANSFORMADORES, MAPA_PLOT,
    OUTPUT_SARIMAX, OUTPUT_UV, OUTPUT_MV, OUTPUT_DL, OUTPUT_SVR,
    OUTPUT_REPORT, N_DL_RUNS
)
from utils import log

OUTPUT_STATS = "resultados_stats"


def carregar_metricas_pontuais():
    """
    Lê os CSVs de métricas dos modelos não-DL e do best-run dos DL.
    Retorna DF com colunas: transformador, modelo, abordagem,
                             RMSE, MAE, sMAPE, R2, n_features
    """
    fontes = [
        (f"{OUTPUT_SARIMAX}/metricas_SARIMAX.csv", "Statistical"),
        (f"{OUTPUT_UV}/metricas_UV.csv",            None),
        (f"{OUTPUT_MV}/metricas_MV.csv",            None),
        (f"{OUTPUT_SVR}/metricas_SVR.csv",          None),
        (f"{OUTPUT_DL}/metricas_DL.csv",            None),
        ("resultados_NAIVE/metricas_SeasonalNaive.csv", None),
    ]

    rows = []
    for path, default_mode in fontes:
        if not os.path.exists(path):
            log.warning(f"Arquivo não encontrado: {path}")
            continue
        df = pd.read_csv(path)
        if default_mode is not None and "abordagem" not in df.columns:
            df["abordagem"] = default_mode
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def carregar_dl_runs():
    """
    Lê todos os CSVs runs_detail_*.csv e calcula (média ± dp) por arch×mode×tid.
    """
    rows = []
    for f in glob.glob(f"{OUTPUT_DL}/runs_detail_*.csv"):
        base = os.path.basename(f).replace("runs_detail_", "").replace(".csv", "")
        try:
            tid, arch, mode = base.rsplit("_", 2)
        except ValueError:
            log.warning(f"Nome de arquivo não parseável: {f}")
            continue
        df_runs = pd.read_csv(f)
        if df_runs.empty:
            continue
        rows.append({
            "transformador": tid,
            "modelo":        arch,
            "abordagem":     mode,
            "n_runs":        len(df_runs),
            "RMSE_mean":     df_runs["RMSE"].mean(),
            "RMSE_std":      df_runs["RMSE"].std(ddof=1),
            "MAE_mean":      df_runs["MAE"].mean(),
            "MAE_std":       df_runs["MAE"].std(ddof=1),
            "sMAPE_mean":    df_runs["sMAPE"].mean(),
            "sMAPE_std":     df_runs["sMAPE"].std(ddof=1),
            "R2_mean":       df_runs["R2"].mean(),
            "R2_std":        df_runs["R2"].std(ddof=1),
        })
    return pd.DataFrame(rows)


def construir_tabela7():
    """
    Constrói a Tabela 7 final.
    """
    metr = carregar_metricas_pontuais()
    if metr.empty:
        log.error("Nenhuma métrica pontual encontrada.")
        return None

    dl_runs = carregar_dl_runs()

    # ── Bootstrap CI ──────────────────────────────────────────────────────
    bs_path = os.path.join(OUTPUT_STATS, "bootstrap_ci_test_metrics.csv")
    bs = pd.read_csv(bs_path) if os.path.exists(bs_path) else pd.DataFrame()

    # ── Diebold-Mariano par-a-par ─────────────────────────────────────────
    dm_path = os.path.join(OUTPUT_STATS, "diebold_mariano_pairs.csv")
    dm = pd.read_csv(dm_path) if os.path.exists(dm_path) else pd.DataFrame()

    # ── Junta tudo ────────────────────────────────────────────────────────
    out_rows = []
    for tid in TRANSFORMADORES:
        sub = metr[metr["transformador"] == tid].copy()
        if sub.empty:
            continue

        for _, row in sub.iterrows():
            modelo = row["modelo"]
            abord  = row.get("abordagem", "n/a")

            rec = {
                "transformador": tid,
                "nome_plot":     MAPA_PLOT.get(tid, tid),
                "modelo":        modelo,
                "abordagem":     abord,
                "RMSE":          row.get("RMSE", np.nan),
                "MAE":           row.get("MAE", np.nan),
                "sMAPE":         row.get("sMAPE", np.nan),
                "R2":            row.get("R2", np.nan),
            }

            # DL runs (média ± dp)
            if not dl_runs.empty:
                m_dl = dl_runs[(dl_runs["transformador"] == tid) &
                                (dl_runs["modelo"] == modelo) &
                                (dl_runs["abordagem"] == abord)]
                if not m_dl.empty:
                    rec["RMSE_mean"]  = round(float(m_dl["RMSE_mean"].iloc[0]),  4)
                    rec["RMSE_std"]   = round(float(m_dl["RMSE_std"].iloc[0]),   4)
                    rec["sMAPE_mean"] = round(float(m_dl["sMAPE_mean"].iloc[0]), 4)
                    rec["sMAPE_std"]  = round(float(m_dl["sMAPE_std"].iloc[0]),  4)
                    rec["R2_mean"]    = round(float(m_dl["R2_mean"].iloc[0]),    4)
                    rec["R2_std"]     = round(float(m_dl["R2_std"].iloc[0]),     4)
                    rec["n_runs"]     = int(m_dl["n_runs"].iloc[0])
                else:
                    rec["n_runs"] = 1
            else:
                rec["n_runs"] = 1

            # Bootstrap CI (sempre disponível, vem de stats_tests.py)
            if not bs.empty:
                m_bs = bs[(bs["transformador"] == tid) &
                          (bs["modelo"].str.startswith(f"{modelo}_{abord}") |
                           (bs["modelo"] == modelo))]
                if not m_bs.empty:
                    rec["RMSE_lower95"] = round(float(m_bs["RMSE_lower95"].iloc[0]), 4)
                    rec["RMSE_upper95"] = round(float(m_bs["RMSE_upper95"].iloc[0]), 4)

            out_rows.append(rec)

    df = pd.DataFrame(out_rows).sort_values(["transformador", "RMSE"])

    # ── Marca o melhor de cada transformador e p-value DM contra o melhor ──
    df["is_best"]      = False
    df["dm_p_vs_best"] = np.nan
    df["sig_diff_vs_best"] = None  # True se p<0.05; False se não significante

    for tid in df["transformador"].unique():
        idx_tid = df[df["transformador"] == tid].index
        if len(idx_tid) == 0:
            continue
        best_idx = df.loc[idx_tid, "RMSE"].idxmin()
        df.loc[best_idx, "is_best"] = True

        if dm.empty:
            continue

        best_modelo_label = f"{df.loc[best_idx, 'modelo']}_{df.loc[best_idx, 'abordagem']}"
        for i in idx_tid:
            if i == best_idx:
                df.loc[i, "dm_p_vs_best"] = 1.0
                df.loc[i, "sig_diff_vs_best"] = False
                continue
            modelo_label = f"{df.loc[i, 'modelo']}_{df.loc[i, 'abordagem']}"
            mask = ((dm["transformador"] == tid) &
                    (((dm["modelo_1"] == best_modelo_label) & (dm["modelo_2"] == modelo_label)) |
                     ((dm["modelo_2"] == best_modelo_label) & (dm["modelo_1"] == modelo_label))))
            sel = dm[mask]
            if not sel.empty:
                p = float(sel["p_value"].iloc[0])
                df.loc[i, "dm_p_vs_best"] = round(p, 4)
                df.loc[i, "sig_diff_vs_best"] = (p < 0.05)

    # ── Salvar CSV ────────────────────────────────────────────────────────
    out_csv = os.path.join(OUTPUT_REPORT, "table7_final.csv")
    df.to_csv(out_csv, index=False)
    log.info(f"\n[#15] Tabela 7 final salva em {out_csv}")

    # ── Gerar versão LaTeX para colar no paper ────────────────────────────
    gerar_latex(df)

    print(df[["transformador", "modelo", "abordagem", "RMSE", "sMAPE",
              "R2", "is_best", "dm_p_vs_best"]].to_string(index=False))
    return df


def gerar_latex(df: pd.DataFrame):
    """
    Gera versão LaTeX da Tabela 7 com:
      - DL: RMSE_mean ± RMSE_std
      - Demais: RMSE pontual
      - Negrito em todos os modelos NÃO significativamente piores que o melhor
      - p-value DM na última coluna
    """
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\caption{Complete forecasting results. RMSE, MAE, sMAPE and $R^2$ "
                 r"on the 2024 test set. Deep learning rows report mean $\pm$ standard "
                 r"deviation across " + str(N_DL_RUNS) + r" runs with distinct seeds; other rows are "
                 r"point estimates. Bold marks the model with lowest RMSE per "
                 r"transformer plus all other models not significantly worse "
                 r"(Diebold-Mariano $p > 0.05$).}")
    lines.append(r"\label{tab:complete_results}")
    lines.append(r"\begin{tabular}{l l c c c c r}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Config.} & \textbf{RMSE} & "
                 r"\textbf{MAE} & \textbf{sMAPE (\%)} & $\mathbf{R^2}$ & "
                 r"\textbf{DM $p$ vs. best} \\")
    lines.append(r"\midrule")

    for tid in df["transformador"].unique():
        nome_plot = MAPA_PLOT.get(tid, tid)
        sub = df[df["transformador"] == tid].copy()
        if sub.empty:
            continue
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{nome_plot}}}}} \\")

        for _, r in sub.iterrows():
            # negrito: melhor OU não significativamente pior
            in_bold = bool(r["is_best"]) or (
                r["sig_diff_vs_best"] is False and not pd.isna(r.get("dm_p_vs_best"))
            )
            fmt = (lambda x: f"\\textbf{{{x}}}") if in_bold else (lambda x: x)

            # RMSE: usa mean±std se houver runs múltiplos, senão pontual
            if not pd.isna(r.get("RMSE_std")) and r.get("n_runs", 1) > 1:
                rmse_str = f"{r['RMSE_mean']:.3f} $\\pm$ {r['RMSE_std']:.3f}"
            else:
                rmse_str = f"{r['RMSE']:.3f}"

            sm = (f"{r.get('sMAPE_mean', r['sMAPE']):.2f}"
                  if not pd.isna(r.get('sMAPE_mean')) and r.get('n_runs', 1) > 1
                  else f"{r['sMAPE']:.2f}")
            r2_disp = (f"{r.get('R2_mean', r['R2']):.3f}"
                       if not pd.isna(r.get('R2_mean')) and r.get('n_runs', 1) > 1
                       else f"{r['R2']:.3f}")

            p_str = ("---" if r["is_best"] else
                     ("---" if pd.isna(r.get("dm_p_vs_best"))
                      else f"{r['dm_p_vs_best']:.3f}"))

            mae_str = f"{r['MAE']:.3f}"

            lines.append(
                f"{fmt(r['modelo'])} & {fmt(r['abordagem'])} & "
                f"{fmt(rmse_str)} & {fmt(mae_str)} & "
                f"{fmt(sm)} & {fmt(r2_disp)} & {p_str} \\\\"
            )
        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out_tex = os.path.join(OUTPUT_REPORT, "table7_final.tex")
    with open(out_tex, "w") as f:
        f.write("\n".join(lines))
    log.info(f"[#15] Versão LaTeX salva em {out_tex}")


if __name__ == "__main__":
    construir_tabela7()
