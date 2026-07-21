# =============================================================================
# EDA.py — Exploratory Data Analysis (dissertação de mestrado)
# Transformer-level load forecasting — João Pessoa, Brazil
# =============================================================================
# Standalone: não depende de utils.py externo.
# Compatível com config.py e estrutura real do projeto.
#
# Seções geradas:
#   01  Série temporal completa com MA-30 e bandas train/val/test
#   02  Histograma de Smax
#   03  Boxplot de Smax + outliers IQR destacados na série
#   04  Decomposição STL semanal   (period=7)
#   05  Decomposição STL anual     (period=365)
#   06  ACF/PACF até lag 400
#   07  ACF nos lags selecionados do modelo (barchart)
#   08  Boxplots por dia da semana e mês
#   09  Perfis de demanda (mean±std) por dia, mês e ano
#   10  Heatmap Year × Day-of-Year
#   11  Testes de estacionariedade ADF + KPSS (janela de treino)
#   12  Codificação cíclica sin/cos (figura explicativa)
#   13  Rolling mean 365 dias causal como feature de tendência
#   14  Scatter Smax vs variáveis climáticas + r de Pearson
#   15  Heatmap Year × Month  +  Matriz de correlação Smax × clima
#   16  Análise comparativa entre os 3 transformadores
#   17  Perfil semanal comparativo (3 transformadores)
#   18  Tabela de autocorrelações nos lags do modelo (CSV)
#   19  Estatísticas descritivas (CSV)
#   20  Testes de estacionariedade (CSV)
# =============================================================================

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.stats as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO — espelha config.py sem importar (standalone)
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH    = "bases_tratadas/daily_peak_transformers_dataset.csv"
WEATHER_PATH = "bases_tratadas/weather_dataset_2022-2024.csv"
USE_WEATHER  = os.path.exists(WEATHER_PATH)

TRANSFORMERS = ["T21a", "T22a", "T70a"]

MAPA_PLOT = {
    "T21a": "T1 (Comercial)",
    "T22a": "T2 (Residencial)",
    "T70a": "T3 (Costeiro/Misto)",
}

START_DATE     = "2021-01-01"
END_TRAIN      = "2022-12-31"
END_VALIDATION = "2023-12-31"
# tudo após END_VALIDATION → teste (2024)

MODEL_LAGS = [1, 2, 7, 14, 30, 60, 180, 365]
PERIOD     = 365

OUT_DIR = "eda_out"
os.makedirs(OUT_DIR, exist_ok=True)

# Paleta de cores por transformador
COLORS = {
    "T21a": "#2563eb",   # azul  → T1 Comercial
    "T22a": "#16a34a",   # verde → T2 Residencial
    "T70a": "#dc2626",   # vermelho → T3 Costeiro
}

# ─────────────────────────────────────────────────────────────────────────────
# ESTILO GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         12,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   11,
    "figure.titlesize":  14,
    "axes.titlepad":     8,
    "figure.dpi":        160,
    "savefig.dpi":       160,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def savefig(path: str, dpi: int = 160):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")


def nome(tid: str) -> str:
    return MAPA_PLOT.get(tid, tid)


def out(tid: str, suffix: str) -> str:
    return os.path.join(OUT_DIR, f"{tid}_{suffix}")


def preparar_serie(df_tr: pd.DataFrame) -> pd.Series:
    s = df_tr.copy()
    time_col = "date" if "date" in s.columns else "datahora"
    s["Time"] = pd.to_datetime(s[time_col])
    s = s.set_index("Time").sort_index()
    full_idx = pd.date_range(s.index.min().normalize(),
                             s.index.max().normalize(), freq="D")
    s = s.reindex(full_idx)
    s = s.loc[pd.to_datetime(START_DATE):]
    s["Smax"] = s["Smax"].interpolate(method="time", limit=7).ffill(limit=2)
    return s["Smax"]


def load_weather() -> pd.DataFrame:
    w = pd.read_csv(WEATHER_PATH, sep=";", encoding="latin-1")
    # Suporte a coluna DATA ou date
    date_col = "DATA" if "DATA" in w.columns else "date"
    w[date_col] = pd.to_datetime(w[date_col])
    w = w.set_index(date_col).sort_index()
    rename = {
        "temp_max_anos":     "temp_max_hist",
        "temp_min_anos":     "temp_min_hist",
        "temp_media_anos":   "temp_mean_hist",
        "precipitacao_anos": "precip_hist",
    }
    cols = [c for c in rename if c in w.columns]
    return w[cols].rename(columns=rename).copy()


def get_splits(y: pd.Series) -> dict:
    t = pd.to_datetime(END_TRAIN)
    v = pd.to_datetime(END_VALIDATION)
    return dict(
        train=y.index[y.index <= t],
        val=y.index[(y.index > t) & (y.index <= v)],
        test=y.index[y.index > v],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 01. SÉRIE TEMPORAL COMPLETA COM BANDAS E MA-30
# ─────────────────────────────────────────────────────────────────────────────
def plot_series_overview(y: pd.Series, tid: str):
    sp = get_splits(y)
    color = COLORS.get(tid, "#2563eb")

    fig, ax = plt.subplots(figsize=(14, 4))
    # Bandas de período
    ax.axvspan(sp["train"][0],  sp["train"][-1],  alpha=0.07, color="#2563eb", label="Train")
    ax.axvspan(sp["val"][0],    sp["val"][-1],    alpha=0.07, color="#f59e0b", label="Validation")
    ax.axvspan(sp["test"][0],   sp["test"][-1],   alpha=0.07, color="#16a34a", label="Test")
    ax.axvline(pd.to_datetime(END_TRAIN),      color="gray", ls=":", lw=1.2)
    ax.axvline(pd.to_datetime(END_VALIDATION), color="gray", ls=":", lw=1.2)
    # Série + MA
    ax.plot(y.index, y.values, lw=0.6, color=color, alpha=0.55)
    ax.plot(y.rolling(30).mean(), lw=2.2, color=color, label="30-day MA")
    ax.set_title(f"{nome(tid)} — Daily Peak Apparent Power ($S_{{\\max}}$)")
    ax.set_xlabel("Date"); ax.set_ylabel("$S_{\\max}$ (normalized)")
    ax.legend(fontsize=10, ncol=4, loc="upper left")
    savefig(out(tid, "01_series_overview.svg"))
    print(f"  [{tid}] 01_series_overview saved")


# ─────────────────────────────────────────────────────────────────────────────
# 02. HISTOGRAMA
# ─────────────────────────────────────────────────────────────────────────────
def plot_histogram(y: pd.Series, tid: str):
    color = COLORS.get(tid, "#2563eb")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(y.dropna().values, bins=60, color=color, alpha=0.85, edgecolor="white")
    ax.axvline(y.mean(),   color="#dc2626", ls="--", lw=1.8, label=f"Mean={y.mean():.3f}")
    ax.axvline(y.median(), color="#f59e0b", ls="--", lw=1.8, label=f"Median={y.median():.3f}")
    ax.set_title(f"{nome(tid)} — Distribution of $S_{{\\max}}$")
    ax.set_xlabel("$S_{\\max}$ (normalized)"); ax.set_ylabel("Count")
    ax.legend()
    savefig(out(tid, "02_histogram.svg"))
    print(f"  [{tid}] 02_histogram saved")


# ─────────────────────────────────────────────────────────────────────────────
# 03. BOXPLOT + OUTLIERS DESTACADOS NA SÉRIE
# ─────────────────────────────────────────────────────────────────────────────
def plot_outlier_analysis(y: pd.Series, tid: str) -> dict:
    Q1, Q3 = y.quantile(0.25), y.quantile(0.75)
    IQR    = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = y[(y < lower) | (y > upper)]
    pct = 100 * len(outliers) / len(y)
    color = COLORS.get(tid, "#2563eb")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Boxplot
    axes[0].boxplot(y.dropna(), vert=True, patch_artist=True, widths=0.5,
                    boxprops=dict(facecolor=color, alpha=0.35, color=color),
                    medianprops=dict(color="#dc2626", lw=2.5),
                    whiskerprops=dict(color=color), capprops=dict(color=color),
                    flierprops=dict(marker="o", color="#dc2626", markersize=4, alpha=0.5))
    axes[0].set_title(f"{nome(tid)} — Boxplot (IQR method)")
    axes[0].set_ylabel("$S_{\\max}$ (normalized)")
    axes[0].set_xticks([1]); axes[0].set_xticklabels([nome(tid)])

    # Série com outliers
    axes[1].plot(y.index, y.values, lw=0.6, color=color, alpha=0.55, label="$S_{\\max}$")
    if len(outliers):
        axes[1].scatter(outliers.index, outliers.values, color="#dc2626",
                        s=20, zorder=5, label=f"Outliers ({pct:.1f}%)")
    axes[1].axhline(upper, ls="--", color="#f59e0b", lw=1.2, label="IQR bounds")
    axes[1].axhline(lower, ls="--", color="#f59e0b", lw=1.2)
    axes[1].set_title(f"{nome(tid)} — Outliers over Time")
    axes[1].set_xlabel("Date"); axes[1].set_ylabel("$S_{\\max}$ (normalized)")
    axes[1].legend(fontsize=10, loc="upper left")

    savefig(out(tid, "03_outliers.svg"))
    print(f"  [{tid}] 03_outliers saved  (outlier rate={pct:.1f}%)")
    return {"n_outliers": len(outliers), "outlier_pct": round(pct, 2),
            "IQR": round(float(IQR), 4), "lower": round(float(lower), 4),
            "upper": round(float(upper), 4)}


# ─────────────────────────────────────────────────────────────────────────────
# 04 + 05. DECOMPOSIÇÃO STL (semanal e anual)
# ─────────────────────────────────────────────────────────────────────────────
def plot_stl(y: pd.Series, tid: str, period: int, label: str, fig_num: str):
    result = STL(y, period=period, robust=True).fit()
    var_resid = result.resid.var()
    var_obs   = y.var()
    frac = 100 * var_resid / var_obs if var_obs > 0 else 0.0
    color = COLORS.get(tid, "#2563eb")

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    pairs = [
        (y,               "Observed",             color),
        (result.trend,    "Trend",                "#16a34a"),
        (result.seasonal, f"Seasonal ({label})",  "#f59e0b"),
        (result.resid,    f"Residual  (σ²={frac:.1f}% of total)", "#6366f1"),
    ]
    for ax, (data, lbl, c) in zip(axes, pairs):
        ax.plot(data.index, data.values, lw=0.85, color=c)
        ax.set_ylabel(lbl, fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
    axes[3].axhline(0, color="red", ls="--", lw=1)
    axes[0].set_title(f"{nome(tid)} — STL Decomposition ({label}, period={period})")
    axes[-1].set_xlabel("Date")
    savefig(out(tid, f"{fig_num}_stl_{label.replace(' ', '_')}.svg"))
    print(f"  [{tid}] {fig_num}_stl_{label} saved  (residual var={frac:.1f}%)")
    return round(frac, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 06. ACF / PACF ATÉ LAG 400
# ─────────────────────────────────────────────────────────────────────────────
def plot_acf_pacf_full(y: pd.Series, tid: str, nlags: int = 400):
    color = COLORS.get(tid, "#2563eb")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    plot_acf( y.dropna(), ax=axes[0], lags=nlags, alpha=0.05, color=color)
    plot_pacf(y.dropna(), ax=axes[1], lags=nlags, alpha=0.05,
              method="ywm", color=color)

    for ax, lbl in zip(axes, ["ACF", "PACF"]):
        ax.set_title(f"{nome(tid)} — {lbl} (up to lag {nlags})")
        ax.set_xlabel("Lag (days)")
        # Marcar lags 7 e 365
        ax.axvline(7,   color="#f59e0b", ls="--", lw=0.9, alpha=0.7, label="lag 7")
        ax.axvline(365, color="#dc2626", ls="--", lw=0.9, alpha=0.7, label="lag 365")
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=10)

    savefig(out(tid, "06_acf_pacf_full.svg"))
    print(f"  [{tid}] 06_acf_pacf_full (lag {nlags}) saved")


# ─────────────────────────────────────────────────────────────────────────────
# 07. ACF NOS LAGS DO MODELO (barchart)
# ─────────────────────────────────────────────────────────────────────────────
def plot_acf_model_lags(y: pd.Series, tid: str) -> dict:
    lags_all = [1, 2, 7, 14, 30, 60, 90, 180, 270, 365]
    acf_vals = [y.autocorr(lag=l) for l in lags_all]
    color    = COLORS.get(tid, "#2563eb")

    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.bar([str(l) for l in lags_all], acf_vals,
                  color=color, alpha=0.75, edgecolor="white")

    # Destacar os lags que entram no modelo
    model_set = set(MODEL_LAGS)
    for bar, lag, val in zip(bars, lags_all, acf_vals):
        if lag in model_set:
            bar.set_edgecolor("#dc2626"); bar.set_linewidth(2.2)
        ypos = val + 0.015 if val >= 0 else val - 0.04
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:.2f}", ha="center", fontsize=8.5)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"{nome(tid)} — Autocorrelation at Key Lags\n"
                 f"(red border = model feature lags: {MODEL_LAGS})")
    savefig(out(tid, "07_acf_model_lags.svg"))
    print(f"  [{tid}] 07_acf_model_lags saved")
    return {f"acf_lag{l}": round(v, 4) for l, v in zip(lags_all, acf_vals)}


# ─────────────────────────────────────────────────────────────────────────────
# 08. BOXPLOTS SAZONAIS (dia da semana e mês)
# ─────────────────────────────────────────────────────────────────────────────
def plot_seasonal_boxplots(y: pd.Series, tid: str):
    df = pd.DataFrame({"Smax": y,
                       "weekday": y.index.weekday,
                       "month":   y.index.month})
    color = COLORS.get(tid, "#2563eb")
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    mon_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, xlabels, title in [
        (axes[0], "weekday", day_labels, "Smax by Weekday"),
        (axes[1], "month",   mon_labels, "Smax by Month"),
    ]:
        groups = [df.loc[df[col] == v, "Smax"].dropna().values
                  for v in sorted(df[col].unique())]
        ax.boxplot(groups, patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.35),
                   medianprops=dict(color="#dc2626", lw=2),
                   whiskerprops=dict(color=color), capprops=dict(color=color),
                   flierprops=dict(marker=".", color=color, markersize=3, alpha=0.4))
        ax.set_xticks(range(1, len(xlabels) + 1))
        ax.set_xticklabels(xlabels, fontsize=10)
        ax.set_title(f"{nome(tid)} — {title}")
        ax.set_ylabel("$S_{\\max}$ (normalized)")
        if col == "weekday":
            ax.axvspan(5.5, 7.5, alpha=0.08, color="#f59e0b")
    plt.suptitle("")
    savefig(out(tid, "08_seasonal_boxplots.svg"))
    print(f"  [{tid}] 08_seasonal_boxplots saved")


# ─────────────────────────────────────────────────────────────────────────────
# 09. PERFIS DE DEMANDA (mean ± std)
# ─────────────────────────────────────────────────────────────────────────────
def plot_demand_profiles(y: pd.Series, tid: str):
    df = pd.DataFrame({"Smax": y,
                       "weekday": y.index.weekday,
                       "month":   y.index.month,
                       "year":    y.index.year})
    color = COLORS.get(tid, "#2563eb")
    day_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    mon_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Dia da semana
    wd = df.groupby("weekday")["Smax"].agg(["mean", "std"])
    axes[0].bar(day_labels, wd["mean"], color=color, alpha=0.8)
    axes[0].errorbar(day_labels, wd["mean"], yerr=wd["std"],
                     fmt="none", color="#dc2626", capsize=4, lw=1.5)
    axes[0].axvspan(4.5, 6.5, alpha=0.1, color="#f59e0b")
    axes[0].set_title(f"{nome(tid)} — Mean $S_{{\\max}}$ by Weekday")
    axes[0].set_ylabel("$S_{\\max}$ (normalized)")

    # Mês
    mo = df.groupby("month")["Smax"].agg(["mean", "std"])
    axes[1].bar(mon_labels, mo["mean"], color=color, alpha=0.8)
    axes[1].errorbar(mon_labels, mo["mean"], yerr=mo["std"],
                     fmt="none", color="#dc2626", capsize=4, lw=1.5)
    axes[1].set_title(f"{nome(tid)} — Mean $S_{{\\max}}$ by Month")
    axes[1].set_ylabel("$S_{\\max}$ (normalized)")
    axes[1].tick_params(axis="x", rotation=45)

    # Tendência anual
    yr = df.groupby("year")["Smax"].agg(["mean", "std"])
    axes[2].bar(yr.index.astype(str), yr["mean"], color=color, alpha=0.8)
    axes[2].errorbar(yr.index.astype(str), yr["mean"], yerr=yr["std"],
                     fmt="none", color="#dc2626", capsize=4, lw=1.5)
    axes[2].set_title(f"{nome(tid)} — Annual Mean $S_{{\\max}}$ (Growth Trend)")
    axes[2].set_ylabel("$S_{\\max}$ (normalized)")

    plt.suptitle(f"{nome(tid)} — Demand Profiles (mean ± 1 std)", y=1.01, fontsize=13)
    savefig(out(tid, "09_demand_profiles.svg"))
    print(f"  [{tid}] 09_demand_profiles saved")


# ─────────────────────────────────────────────────────────────────────────────
# 10. HEATMAP YEAR × DAY-OF-YEAR
# ─────────────────────────────────────────────────────────────────────────────
def plot_year_doy_heatmap(y: pd.Series, tid: str):
    df = pd.DataFrame({"Smax": y, "year": y.index.year, "doy": y.index.dayofyear})
    pivot = df.pivot_table(index="year", columns="doy", values="Smax", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(15, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                   extent=[1, 366, pivot.index[-1] + 0.5, pivot.index[0] - 0.5])
    ax.set_title(f"{nome(tid)} — Heatmap: Year × Day-of-Year")
    ax.set_xlabel("Day of Year"); ax.set_ylabel("Year")
    ax.set_yticks(pivot.index)
    ax.set_yticklabels(pivot.index, fontsize=10)
    # Marcar início de cada mês
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names  = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_names, fontsize=9)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("$S_{\\max}$ (normalized)", fontsize=11)
    savefig(out(tid, "10_heatmap_year_doy.svg"))
    print(f"  [{tid}] 10_heatmap_year_doy saved")


# ─────────────────────────────────────────────────────────────────────────────
# 11. TESTES DE ESTACIONARIEDADE (ADF + KPSS — janela de treino)
# ─────────────────────────────────────────────────────────────────────────────
def stationarity_tests(y: pd.Series, tid: str) -> dict:
    sp = get_splits(y)
    y_train = y.loc[sp["train"]].dropna()

    adf_res  = adfuller(y_train, autolag="AIC")
    adf_stat, adf_p = float(adf_res[0]), float(adf_res[1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_res = kpss(y_train, regression="c", nlags="auto")
    kpss_stat, kpss_p = float(kpss_res[0]), float(kpss_res[1])

    if   adf_p < 0.05 and kpss_p > 0.05:
        conclusion = "stationary"
    elif adf_p >= 0.05 and kpss_p <= 0.05:
        conclusion = "non-stationary"
    else:
        conclusion = "borderline"

    result = {
        "transformer":  nome(tid),
        "id":           tid,
        "n_train_days": int(len(y_train)),
        "adf_stat":     round(adf_stat,  3),
        "adf_p":        round(adf_p,     4),
        "kpss_stat":    round(kpss_stat, 3),
        "kpss_p":       round(kpss_p,    4),
        "conclusion":   conclusion,
    }
    print(f"  [{tid}] ADF: {adf_stat:.3f} (p={adf_p:.4f}) | "
          f"KPSS: {kpss_stat:.3f} (p={kpss_p:.4f}) → {conclusion}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 12. CODIFICAÇÃO CÍCLICA SIN/COS (figura explicativa — uma vez)
# ─────────────────────────────────────────────────────────────────────────────
def plot_cyclic_encoding():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: codificação cíclica — dias da semana
    day_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    days = np.arange(7)
    sin_w = np.sin(2 * np.pi * days / 7)
    cos_w = np.cos(2 * np.pi * days / 7)
    sc = axes[0].scatter(sin_w, cos_w, c=days, cmap="hsv", s=130, zorder=5, edgecolors="k")
    for i, (x, y_, lbl) in enumerate(zip(sin_w, cos_w, day_labels)):
        axes[0].annotate(lbl, (x, y_), textcoords="offset points", xytext=(6, 4), fontsize=9)
    axes[0].set_title("Cyclic Encoding — Weekday\n(Mon ≈ Sun on the circle)")
    axes[0].set_xlabel("sin(weekday × 2π/7)"); axes[0].set_ylabel("cos(weekday × 2π/7)")
    axes[0].set_aspect("equal")
    plt.colorbar(sc, ax=axes[0], label="Weekday index")

    # Panel B: codificação linear — problema
    axes[1].plot(days, days, "o-", color="#2563eb", lw=2)
    axes[1].set_xticks(days); axes[1].set_xticklabels(day_labels, fontsize=9)
    axes[1].set_title("Linear Encoding — Weekday\n(Sun=6 ≠ Mon=0: artificial gap)")
    axes[1].set_xlabel("Weekday index"); axes[1].set_ylabel("Encoded value")
    axes[1].annotate("Artificial\ndiscontinuity", xy=(6, 6), xytext=(3.5, 5.5),
                     arrowprops=dict(arrowstyle="->", color="#dc2626"),
                     color="#dc2626", fontsize=10)

    # Panel C: dia do ano
    doy = np.arange(1, 366)
    sin_d = np.sin(2 * np.pi * doy / 365)
    cos_d = np.cos(2 * np.pi * doy / 365)
    sc2 = axes[2].scatter(sin_d, cos_d, c=doy, cmap="hsv", s=5, alpha=0.85)
    axes[2].set_title("Cyclic Encoding — Day of Year\n(Dec 31 ≈ Jan 1)")
    axes[2].set_xlabel("sin(doy × 2π/365)"); axes[2].set_ylabel("cos(doy × 2π/365)")
    axes[2].set_aspect("equal")
    plt.colorbar(sc2, ax=axes[2], label="Day of Year")

    plt.suptitle("Cyclic (sin/cos) Feature Encoding — Rationale", y=1.02, fontsize=14)
    savefig(os.path.join(OUT_DIR, "00_cyclic_encoding.svg"))
    print("  [global] 00_cyclic_encoding saved")


# ─────────────────────────────────────────────────────────────────────────────
# 13. ROLLING MEAN 365d CAUSAL (feature de tendência)
# ─────────────────────────────────────────────────────────────────────────────
def plot_rolling_mean_feature(y: pd.Series, tid: str):
    rolling  = y.shift(1).rolling(365, min_periods=180).mean()
    residual = y - rolling
    color = COLORS.get(tid, "#2563eb")

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(y.index, y.values, lw=0.5, color=color, alpha=0.5, label="$S_{\\max}$")
    axes[0].plot(rolling.index, rolling.values, lw=2.2, color=color, label="365-day causal MA")
    axes[0].set_title(f"{nome(tid)} — 365-day Causal Rolling Mean (model feature, no leakage)")
    axes[0].set_ylabel("$S_{\\max}$ (normalized)"); axes[0].legend()

    axes[1].plot(residual.index, residual.values, lw=0.7, color="#6366f1", alpha=0.8)
    axes[1].axhline(0, color="black", lw=0.9, ls="--")
    axes[1].set_title("Residual after removing rolling mean")
    axes[1].set_xlabel("Date"); axes[1].set_ylabel("Residual")

    savefig(out(tid, "13_rolling_mean_feature.svg"))
    print(f"  [{tid}] 13_rolling_mean_feature saved")


# ─────────────────────────────────────────────────────────────────────────────
# 14. SCATTER Smax × VARIÁVEIS CLIMÁTICAS + r de Pearson
# ─────────────────────────────────────────────────────────────────────────────
def plot_weather_scatter(y: pd.Series, weather: pd.DataFrame, tid: str):
    df    = pd.DataFrame({"Smax": y}).join(weather, how="left").ffill().bfill().dropna()
    wcols = [c for c in weather.columns if c in df.columns]
    if not wcols:
        print(f"  [{tid}] No weather columns — skipping scatter")
        return

    labels_map = {
        "temp_mean_hist": "Mean Temp (°C, clim.)",
        "temp_max_hist":  "Max Temp (°C, clim.)",
        "temp_min_hist":  "Min Temp (°C, clim.)",
        "precip_hist":    "Precipitation (mm, clim.)",
    }
    n = len(wcols)
    color = COLORS.get(tid, "#2563eb")
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1: axes = [axes]

    for ax, col in zip(axes, wcols):
        ax.scatter(df[col], df["Smax"], alpha=0.25, s=8, color=color)
        r, p   = sp.pearsonr(df[col].values, df["Smax"].values)
        m, b   = np.polyfit(df[col].values, df["Smax"].values, 1)
        xline  = np.linspace(df[col].min(), df[col].max(), 100)
        ax.plot(xline, m * xline + b, color="#dc2626", lw=2)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        lbl = labels_map.get(col, col)
        ax.set_title(f"{nome(tid)}\n{lbl}\nr = {r:.3f} {sig}", fontsize=11)
        ax.set_xlabel(lbl); ax.set_ylabel("$S_{\\max}$ (normalized)")

    plt.suptitle(f"{nome(tid)} — Smax vs Climate Variables", y=1.02, fontsize=13)
    savefig(out(tid, "14_weather_scatter.svg"))
    print(f"  [{tid}] 14_weather_scatter saved")

    # Série duplo eixo: Smax + temperatura
    if "temp_mean_hist" in df.columns:
        fig, ax = plt.subplots(figsize=(14, 4))
        df["Smax"].plot(ax=ax, lw=1.2, color=color, label="$S_{\\max}$")
        ax2 = ax.twinx()
        df["temp_mean_hist"].plot(ax=ax2, lw=1.2, color="#dc2626",
                                  ls="--", label="Mean Temp (clim.)")
        ax.set_title(f"{nome(tid)} — $S_{{\\max}}$ and Climatological Mean Temperature")
        ax.set_xlabel("Date"); ax.set_ylabel("$S_{\\max}$ (normalized)")
        ax2.set_ylabel("Mean Temperature (°C)")
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], fontsize=10, loc="upper left")
        savefig(out(tid, "14b_smax_vs_temp_series.svg"))
        print(f"  [{tid}] 14b_smax_vs_temp_series saved")


# ─────────────────────────────────────────────────────────────────────────────
# 15. HEATMAP YEAR × MONTH  +  MATRIZ DE CORRELAÇÃO
# ─────────────────────────────────────────────────────────────────────────────
def plot_climate_heatmap(y: pd.Series, weather: pd.DataFrame, tid: str):
    try:
        import seaborn as sns
    except ImportError:
        print("  [!] seaborn not installed — skipping climate heatmap (pip install seaborn)")
        return

    df = pd.DataFrame({"Smax": y}).join(weather, how="left").ffill().bfill().dropna()
    df["month"] = df.index.month
    df["year"]  = df.index.year
    numeric_cols = ["Smax"] + [c for c in weather.columns if c in df.columns]
    mon_names = ["Jan","Feb","Mar","Apr","May","Jun",
                 "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Year × Month
    pivot = df.pivot_table(values="Smax", index="year", columns="month", aggfunc="mean")
    pivot.columns = mon_names[:len(pivot.columns)]
    sns.heatmap(pivot, ax=axes[0], cmap="YlOrRd", annot=True, fmt=".2f",
                linewidths=0.4, cbar_kws={"label": "Mean $S_{\\max}$"})
    axes[0].set_title(f"{nome(tid)} — Mean $S_{{\\max}}$ by Year × Month")
    axes[0].set_xlabel("Month"); axes[0].set_ylabel("Year")

    # Correlation matrix (lower triangle)
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, ax=axes[1], cmap="coolwarm", annot=True, fmt=".2f",
                vmin=-1, vmax=1, center=0, linewidths=0.4, mask=mask,
                cbar_kws={"label": "Pearson r"})
    axes[1].set_title(f"{nome(tid)} — Correlation Matrix")

    savefig(out(tid, "15_climate_heatmap.svg"))
    print(f"  [{tid}] 15_climate_heatmap saved")


# ─────────────────────────────────────────────────────────────────────────────
# 16 + 17. ANÁLISE COMPARATIVA ENTRE TRANSFORMADORES
# ─────────────────────────────────────────────────────────────────────────────
def plot_comparative(series_dict: dict):
    """4-panel comparison: MA-30, KDE, CV, annual growth."""
    try:
        from scipy.stats import gaussian_kde
        has_kde = True
    except ImportError:
        has_kde = False

    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    # Panel 1: MA-30 sobreposta
    ax1 = fig.add_subplot(gs[0, 0])
    for tid, y in series_dict.items():
        ax1.plot(y.rolling(30).mean(), lw=2,
                 color=COLORS.get(tid, "#999"), label=nome(tid))
    ax1.set_title("30-day Moving Average — All Transformers")
    ax1.set_ylabel("$S_{\\max}$ (normalized)"); ax1.legend(fontsize=9)

    # Panel 2: KDE das distribuições
    ax2 = fig.add_subplot(gs[0, 1])
    for tid, y in series_dict.items():
        clean = y.dropna().values
        color = COLORS.get(tid, "#999")
        if has_kde:
            kde = gaussian_kde(clean)
            xi  = np.linspace(clean.min(), clean.max(), 300)
            ax2.plot(xi, kde(xi), lw=2.2, color=color, label=nome(tid))
        ax2.axvline(y.mean(), ls="--", color=color, lw=1.2, alpha=0.7)
    ax2.set_title("Demand Distribution (KDE)\n(dashed = mean)")
    ax2.set_xlabel("$S_{\\max}$ (normalized)"); ax2.set_ylabel("Density")
    ax2.legend(fontsize=9)

    # Panel 3: CV comparativo
    ax3 = fig.add_subplot(gs[1, 0])
    tids = list(series_dict.keys())
    cvs  = [series_dict[t].std() / series_dict[t].mean() * 100 for t in tids]
    bars = ax3.bar([nome(t) for t in tids], cvs,
                   color=[COLORS.get(t, "#999") for t in tids], alpha=0.85)
    for bar, v in zip(bars, cvs):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.15, f"{v:.1f}%",
                 ha="center", fontsize=11, fontweight="bold")
    ax3.set_title("Coefficient of Variation (CV %)")
    ax3.set_ylabel("CV (%)")

    # Panel 4: tendência anual
    ax4 = fig.add_subplot(gs[1, 1])
    for tid, y in series_dict.items():
        yr = y.resample("YE").mean()
        ax4.plot(yr.index.year, yr.values, marker="o", lw=2,
                 color=COLORS.get(tid, "#999"), label=nome(tid))
    ax4.set_title("Annual Mean Demand — Growth Trend")
    ax4.set_xlabel("Year"); ax4.set_ylabel("Mean $S_{\\max}$ (normalized)")
    ax4.legend(fontsize=9)

    plt.suptitle("Comparative Analysis — All Transformers", fontsize=15, y=1.01)
    savefig(os.path.join(OUT_DIR, "00_comparative_all.svg"))
    print("  [global] 00_comparative_all saved")


def plot_comparative_weekday(series_dict: dict):
    """Perfil médio semanal — 3 transformadores sobrepostos."""
    fig, ax = plt.subplots(figsize=(10, 5))
    day_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    for tid, y in series_dict.items():
        wd_mean = y.groupby(y.index.weekday).mean()
        ax.plot(day_labels, wd_mean.values, marker="o", lw=2.2,
                color=COLORS.get(tid, "#999"), label=nome(tid))
    ax.axvspan(4.5, 6.5, alpha=0.07, color="#f59e0b", label="Weekend")
    ax.set_title("Mean Weekday Profile — All Transformers Compared")
    ax.set_xlabel("Day of week"); ax.set_ylabel("Mean $S_{\\max}$ (normalized)")
    ax.legend()
    savefig(os.path.join(OUT_DIR, "00_comparative_weekday.svg"))
    print("  [global] 00_comparative_weekday saved")


# ─────────────────────────────────────────────────────────────────────────────
# 18. TABELA DE AUTOCORRELAÇÕES NOS LAGS DO MODELO
# ─────────────────────────────────────────────────────────────────────────────
def build_lag_acf_table(series_dict: dict) -> pd.DataFrame:
    lags_all = [1, 2, 7, 14, 30, 60, 90, 180, 270, 365]
    rows = []
    for tid, y in series_dict.items():
        row = {"Transformer": nome(tid), "ID": tid}
        for lag in lags_all:
            row[f"r(lag={lag})"] = round(y.autocorr(lag=lag), 4)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Transformer")
    df.to_csv(os.path.join(OUT_DIR, "lag_acf_table.csv"))
    print(f"  [global] lag_acf_table.csv saved")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 19. ESTATÍSTICAS DESCRITIVAS COMPLETAS
# ─────────────────────────────────────────────────────────────────────────────
def compute_summary(y: pd.Series, tid: str, outlier_info: dict) -> dict:
    Q1, Q3 = y.quantile(0.25), y.quantile(0.75)
    return {
        "ID":            tid,
        "Transformer":   nome(tid),
        "start":         str(y.index.min().date()),
        "end":           str(y.index.max().date()),
        "n_days":        int(len(y)),
        "mean":          round(float(y.mean()),   4),
        "std":           round(float(y.std()),    4),
        "cv_pct":        round(float(y.std() / y.mean() * 100), 2),
        "min":           round(float(y.min()),    4),
        "q25":           round(float(Q1),         4),
        "median":        round(float(y.median()), 4),
        "q75":           round(float(Q3),         4),
        "max":           round(float(y.max()),    4),
        "IQR":           round(float(Q3 - Q1),    4),
        "skewness":      round(float(y.skew()),   4),
        "kurtosis":      round(float(y.kurt()),   4),
        "missing_pct":   round(float(y.isna().mean() * 100), 2),
        "n_outliers":    outlier_info.get("n_outliers", 0),
        "outlier_pct":   outlier_info.get("outlier_pct", 0.0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  EDA — Transformer Load Forecasting (dissertação)")
    print("=" * 60)

    # Carregar dados
    df = pd.read_csv(DATA_PATH, sep=";", encoding="latin-1")
    df = df[df["id"].isin(TRANSFORMERS)].copy()
    if df.empty:
        raise ValueError(f"Nenhum dado para {TRANSFORMERS}. Verifique DATA_PATH e IDs.")

    weather = None
    if USE_WEATHER:
        try:
            weather = load_weather()
            print(f"  Weather data loaded: {WEATHER_PATH}")
        except Exception as e:
            print(f"  [!] Weather not loaded: {e}")

    all_summaries    = []
    all_stationarity = []
    all_acf          = []
    series_dict      = {}

    # 12. Codificação cíclica (uma vez, não por transformador)
    plot_cyclic_encoding()

    # ── Loop por transformador ─────────────────────────────────────────────
    for tid, df_tr in df.groupby("id"):
        sep = "─" * 55
        print(f"\n{sep}\n  [{tid}] {nome(tid)}\n{sep}")

        y = preparar_serie(df_tr)
        series_dict[tid] = y

        # 01. Série + MA-30 + bandas
        plot_series_overview(y, tid)

        # 02. Histograma
        plot_histogram(y, tid)

        # 03. Outliers
        outlier_info = plot_outlier_analysis(y, tid)

        # 04. STL semanal
        plot_stl(y, tid, period=7,   label="weekly",  fig_num="04")

        # 05. STL anual
        plot_stl(y, tid, period=365, label="annual",  fig_num="05")

        # 06. ACF/PACF completo até lag 400
        plot_acf_pacf_full(y, tid, nlags=400)

        # 07. ACF nos lags do modelo
        acf_row = plot_acf_model_lags(y, tid)
        acf_row.update({"ID": tid, "Transformer": nome(tid)})
        all_acf.append(acf_row)

        # 08. Boxplots sazonais
        plot_seasonal_boxplots(y, tid)

        # 09. Perfis de demanda
        plot_demand_profiles(y, tid)

        # 10. Heatmap Year × DoY
        plot_year_doy_heatmap(y, tid)

        # 11. Testes de estacionariedade
        stat_res = stationarity_tests(y, tid)
        all_stationarity.append(stat_res)

        # 13. Rolling mean causal
        plot_rolling_mean_feature(y, tid)

        # 14. Scatter climático
        if weather is not None:
            plot_weather_scatter(y, weather, tid)

        # 15. Heatmap climático
        if weather is not None:
            plot_climate_heatmap(y, weather, tid)

        # Estatísticas descritivas
        summary = compute_summary(y, tid, outlier_info)
        all_summaries.append(summary)

    # ── Análises globais (múltiplos transformadores) ───────────────────────
    print(f"\n{'─'*55}\n  [global] Comparative analysis\n{'─'*55}")
    if len(series_dict) > 1:
        # 16. Comparativo 4-panel
        plot_comparative(series_dict)
        # 17. Perfil semanal comparativo
        plot_comparative_weekday(series_dict)

    # 18. Tabela de autocorrelações
    lag_table = build_lag_acf_table(series_dict)

    # ── Salvar CSVs finais ─────────────────────────────────────────────────
    pd.DataFrame(all_summaries).to_csv(
        os.path.join(OUT_DIR, "summary_statistics.csv"), index=False)
    pd.DataFrame(all_stationarity).to_csv(
        os.path.join(OUT_DIR, "stationarity_tests.csv"), index=False)

    # ── Imprimir resumo no console ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  EDA COMPLETE — outputs in '{OUT_DIR}/'")
    print(f"{'='*60}")

    print("\n── Descriptive Statistics ────────────────────────────────")
    cols_show = ["Transformer","n_days","mean","std","cv_pct",
                 "skewness","kurtosis","outlier_pct"]
    print(pd.DataFrame(all_summaries)[cols_show].to_string(index=False))

    print("\n── Stationarity Tests (Training Window 2021–2022) ────────")
    cols_stat = ["transformer","n_train_days",
                 "adf_stat","adf_p","kpss_stat","kpss_p","conclusion"]
    print(pd.DataFrame(all_stationarity)[cols_stat].to_string(index=False))

    print("\n── Autocorrelation at Model Lags ─────────────────────────")
    print(lag_table.drop(columns=["ID"], errors="ignore").to_string())


if __name__ == "__main__":
    main()
