# =============================================================================
# models_svr.py — Support Vector Regression (RBF)  ·  UV e MV
# =============================================================================
# >>> REESCRITA (duas correções estruturais) <<<
#
# 1) AVALIAÇÃO RECURSIVA DE VERDADE.
#    A versão anterior avaliava o teste com pipe.predict(X_te), onde X_te era
#    uma matriz de lags construída com os VALORES REAIS de 2024 → previsão de
#    1 passo à frente (teacher forcing), incompatível com o protocolo
#    recursivo de 365 passos aplicado ao boosting/SARIMAX (Paper, Seção 3.5).
#    Agora o SVR usa exatamente a mesma infraestrutura do boosting:
#    ForecasterRecursive + backtesting_forecaster com fold único de 365
#    passos (Fold 1 = HPT na validação; Fold 2 = retreino em treino+val e
#    avaliação única no teste).
#
# 2) UV VERDADEIRAMENTE UNIVARIADO.
#    A versão anterior, no modo "UV", juntava TODOS os atributos calendários
#    (inclusive weekday_sin/cos e a coluna bruta 'year') à matriz de lags —
#    ou seja, o "UV" tinha âncora calendária completa. Agora:
#      UV = lags {1,2,7,14,30,60,180,365} + média móvel de 365 dias (idêntico
#           ao boosting-UV);
#      MV = UV + atributos exógenos selecionados por importância de ganho
#           (τ = 95%, proxy LGBM treinado só no treino) — idêntico ao
#           boosting-MV.
#
# Escalonamento: Pipeline(StandardScaler → SVR) dentro do ForecasterRecursive,
# de modo que o scaler é reajustado a cada fit sem vazamento.
# =============================================================================
import warnings; warnings.filterwarnings("ignore")
import os, random
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.preprocessing import RollingFeatures

from config import (
    TRANSFORMADORES, MAPA_PLOT, RANDOM_STATE,
    LAGS, ROLLING_WINDOW, END_TRAIN, END_VALIDATION,
    OUTPUT_SVR, OUTPUT_FUTURE, FUTURE_END_DATE
)
from utils import (
    preparar_serie_diaria, adicionar_features_calendario,
    carregar_weather, compute_metrics, get_splits,
    plot_forecast_3way, plot_residuals, plot_future_forecast, log
)

# ── Compatibilidade com utils locais que não têm estas funções ──────────────
try:
    from utils import carregar_dados
except ImportError:
    from config import DATA_PATH
    def carregar_dados():
        return pd.read_csv(DATA_PATH, sep=";", encoding="latin-1")

try:
    from utils import seasonal_naive_forecast
except ImportError:
    def seasonal_naive_forecast(y_hist, test_index, period):
        """Naive sazonal recursivo: ŷ_t = y_{t−period} (usa a própria
        previsão quando t−period cai dentro do horizonte)."""
        full, preds = y_hist.copy(), []
        for d in test_index:
            ref = d - pd.Timedelta(days=period)
            val = full.get(ref, np.nan)
            preds.append(val)
            full.loc[d] = val
        return pd.Series(preds, index=test_index)

random.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)

EXOG_COLS = [
    "weekday_sin", "weekday_cos", "month_sin", "month_cos",
    "dayofyear_sin", "dayofyear_cos", "is_weekend", "is_holiday",
    "temp_max_hist", "temp_min_hist", "temp_mean_hist", "precip_hist"
]
FEATURE_IMPORTANCE_THRESHOLD = 0.95

# Grade de HPT do SVR — 6 configurações selecionadas do espaço
# C ∈ {1,10,100,1000} × ε ∈ {0.1,0.5} × γ ∈ {scale,auto}
# (Paper, Tabela 3: "6 selected configurations").
# ATENÇÃO: confirme se estas 6 coincidem com as usadas na versão anterior;
# ajuste a lista se necessário.
SVR_PARAM_GRID = [
    {"svr__C": 1,    "svr__epsilon": 0.1, "svr__gamma": "scale"},
    {"svr__C": 10,   "svr__epsilon": 0.1, "svr__gamma": "scale"},
    {"svr__C": 100,  "svr__epsilon": 0.1, "svr__gamma": "scale"},
    {"svr__C": 1000, "svr__epsilon": 0.1, "svr__gamma": "scale"},
    {"svr__C": 10,   "svr__epsilon": 0.5, "svr__gamma": "scale"},
    {"svr__C": 100,  "svr__epsilon": 0.1, "svr__gamma": "auto"},
]


# ─────────────────────────────────────────────────────────────────────────────
# CONTROLES DE EXECUÇÃO (rerun rápido) — ver models_dl.py para detalhes.
# ─────────────────────────────────────────────────────────────────────────────
RUN_FUTURE = os.environ.get("RUN_FUTURE", "0") == "1"
ONLY_TID   = os.environ.get("ONLY_TID")

class _SkipFuture(Exception):
    pass

# Cache opcional de best_params (preencha a partir de metrics_SVR*.csv,
# coluna best_params, se quiser pular a HPT). Chave: (tid, "SVR", mode).
BEST_PARAMS_CACHE = {}


def make_svr_pipeline() -> Pipeline:
    """StandardScaler + SVR(RBF), reajustados a cada fit do forecaster."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(kernel="rbf")),
    ])


def make_forecaster(params: dict) -> ForecasterRecursive:
    """ForecasterRecursive idêntico em estrutura ao usado no boosting."""
    pipe = make_svr_pipeline().set_params(**params)
    return ForecasterRecursive(
        estimator=pipe, lags=LAGS,
        window_features=RollingFeatures(stats=["mean"],
                                        window_sizes=[ROLLING_WINDOW])
    )


# ─────────────────────────────────────────────────────────────────────────────
# SELEÇÃO DE FEATURES (idêntica ao boosting / DL)
# ─────────────────────────────────────────────────────────────────────────────
def selecionar_features_mv(y_train, exog_train, tid, output_dir):
    proxy = LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        num_leaves=31, random_state=RANDOM_STATE, n_jobs=-1,
        verbose=-1, importance_type="gain"
    )
    fc = ForecasterRecursive(
        estimator=proxy, lags=LAGS,
        window_features=RollingFeatures(stats=["mean"],
                                        window_sizes=[ROLLING_WINDOW])
    )
    fc.fit(y=y_train, exog=exog_train)
    imp_df = fc.get_feature_importances()

    exog_imp = imp_df[imp_df["feature"].isin(exog_train.columns)].copy()
    exog_imp = exog_imp.sort_values("importance", ascending=False).reset_index(drop=True)
    if exog_imp.empty or exog_imp["importance"].sum() == 0:
        return list(exog_train.columns)

    exog_imp["imp_cum"] = (exog_imp["importance"] / exog_imp["importance"].sum()).cumsum()
    selected = exog_imp.loc[
        exog_imp["imp_cum"].shift(1, fill_value=0) < FEATURE_IMPORTANCE_THRESHOLD,
        "feature"
    ].tolist()
    if not selected:
        selected = [exog_imp.iloc[0]["feature"]]

    log.info(f"  [FS-SVR] {tid}: {len(selected)}/{len(exog_imp)} → {selected}")

    colors = ["#2563eb" if f in selected else "#94a3b8" for f in exog_imp["feature"]]
    fig, ax = plt.subplots(figsize=(9, max(4, len(exog_imp) * 0.5 + 1)))
    ax.barh(exog_imp["feature"][::-1], exog_imp["importance"][::-1], color=colors[::-1])
    ax.set_title(f"{MAPA_PLOT.get(tid,tid)} — SVR: Feature Selection\n"
                 f"Blue = selected ({len(selected)}/{len(exog_imp)})", fontsize=13)
    ax.set_xlabel("Importance (gain)", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/feature_selection_{tid}_SVR_MV.svg",
                dpi=150, bbox_inches="tight")
    plt.close()
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def treinar_svr(df: pd.DataFrame, weather=None, mode="UV") -> pd.DataFrame:
    resultados = []

    for tid, df_tr in df.groupby("id"):
        log.info(f"\n{'='*60}\n  {tid}  [SVR-{mode}]\n{'='*60}")
        y      = preparar_serie_diaria(df_tr)
        splits = get_splits(y)

        # ── Seasonal naive baselines (só no modo UV, evita duplicatas) ──
        if mode == "UV":
            y_tv_naive = y.loc[splits["train"].union(splits["val"])]
            for period, label in [(7, "Naive-7d"), (365, "Naive-365d")]:
                naive_pred    = seasonal_naive_forecast(y_tv_naive,
                                                        splits["test"],
                                                        period=period)
                naive_metrics = compute_metrics(y.loc[splits["test"]], naive_pred)
                log.info(f"  {label} → RMSE={naive_metrics['RMSE']:.4f} "
                         f"R²={naive_metrics['R2']:.4f}")
                resultados.append({
                    "transformer": tid, "label": MAPA_PLOT.get(tid, tid),
                    "model": label, "mode": "Naive",
                    "features_sel": "N/A", "n_features_sel": 0,
                    "best_params": "N/A", "MAE_val": np.nan,
                    **{k: round(v, 4) for k, v in naive_metrics.items()},
                })

        # ── Exógenas (apenas MV; UV = série pura) ───────────────────────
        exog              = None
        selected_features = None

        if mode == "MV":
            cal = adicionar_features_calendario(y)
            if weather is not None:
                exog_df = cal.join(weather, how="left").bfill().ffill()
            else:
                exog_df = cal
            cols_disp    = [c for c in EXOG_COLS if c in exog_df.columns]
            exog_full_df = exog_df[cols_disp]

            selected_features = selecionar_features_mv(
                y.loc[splits["train"]],
                exog_full_df.loc[splits["train"]],
                tid, OUTPUT_SVR
            )
            exog = exog_full_df[selected_features]
            log.info(f"  Exogenous features for HPT/train/test: {selected_features}")

        y_tv    = y.loc[splits["train"].union(splits["val"])]
        exog_tv = exog.loc[y_tv.index] if exog is not None else None

        # Fold 1 (HPT): treina 2021-2022, prevê recursivamente 2023 (365 passos)
        cv_val  = TimeSeriesFold(steps=len(splits["val"]),
                                 initial_train_size=len(splits["train"]),
                                 refit=False)
        # Fold 2 (avaliação): treina 2021-2023, prevê recursivamente 2024
        cv_test = TimeSeriesFold(steps=len(splits["test"]),
                                 initial_train_size=len(y_tv),
                                 refit=False)

        # ── HPT na validação (recursivo, como o boosting) ────────────────
        cached = BEST_PARAMS_CACHE.get((tid, "SVR", mode))
        best_mae, best_params = np.inf, None
        for params in (SVR_PARAM_GRID if cached is None else []):
            fc = make_forecaster(params)
            kw = dict(forecaster=fc, y=y_tv, cv=cv_val,
                      metric="mean_absolute_error", verbose=False)
            if exog_tv is not None:
                kw["exog"] = exog_tv
            try:
                _, pv = backtesting_forecaster(**kw)
                mae = mean_absolute_error(y.loc[pv.index], pv["pred"])
            except Exception as e:
                log.warning(f"     {params} → falhou: {e}")
                continue
            log.info(f"     {params} → MAE_val={mae:.4f}")
            if mae < best_mae:
                best_mae, best_params = mae, params

        if cached is not None:
            best_params, best_mae = dict(cached), np.nan
            log.info(f"  ✓ HPT (cache) → {best_params}")

        if best_params is None:
            log.error(f"  {tid}: nenhuma configuração SVR convergiu. Pulando.")
            continue

        log.info(f"  ✓ HPT → {best_params} | MAE_val={best_mae:.4f}")

        # ── Fold 2: retreino em treino+val, previsão recursiva do teste ─
        fc_final = make_forecaster(best_params)
        kw_test = dict(forecaster=fc_final, y=y, cv=cv_test,
                       metric="mean_absolute_error", verbose=False)
        if exog is not None:
            kw_test["exog"] = exog
        _, pt = backtesting_forecaster(**kw_test)

        y_true  = y.loc[pt.index]
        metrics = compute_metrics(y_true, pt["pred"])
        log.info(f"  ✓ TEST → RMSE={metrics['RMSE']:.4f} R²={metrics['R2']:.4f}")

        # Previsão de validação para o plot 3-way (mesmo fold da HPT)
        fc_val_plot = make_forecaster(best_params)
        kw_vp = dict(forecaster=fc_val_plot, y=y_tv, cv=cv_val,
                     metric="mean_absolute_error", verbose=False)
        if exog_tv is not None:
            kw_vp["exog"] = exog_tv
        _, pv_plot = backtesting_forecaster(**kw_vp)

        nome_modelo = f"SVR_{mode}"
        plot_forecast_3way(y, pv_plot["pred"], pt["pred"],
                           tid, "SVR", OUTPUT_SVR, suffix=f"_{mode}")
        plot_residuals(y_true, pt["pred"], tid, nome_modelo, OUTPUT_SVR)

        pd.DataFrame({
            "date":   pt.index,
            "y_true": y_true.values,
            "y_pred": pt["pred"].values,
        }).to_csv(f"{OUTPUT_SVR}/preds_test_{tid}_SVR_{mode}.csv", index=False)

        # ── Previsão futura (retreino em toda a série) ───────────────────
        try:
            if not RUN_FUTURE:
                raise _SkipFuture
            future_idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1),
                                       pd.to_datetime(FUTURE_END_DATE), freq="D")
            fc_fut = make_forecaster(best_params)
            if exog is not None:
                fc_fut.fit(y=y, exog=exog)
                cal_fut = adicionar_features_calendario(
                    pd.Series(np.nan, index=future_idx))
                if weather is not None:
                    for col in [c for c in selected_features
                                if c not in cal_fut.columns and c in weather.columns]:
                        cal_fut[col] = weather[col].mean()
                exog_fut = cal_fut[[c for c in selected_features
                                    if c in cal_fut.columns]]
                preds_fut = fc_fut.predict(steps=len(future_idx), exog=exog_fut)
            else:
                fc_fut.fit(y=y)
                preds_fut = fc_fut.predict(steps=len(future_idx))

            fut_series = pd.Series(preds_fut.values, index=future_idx)

            # IC aproximado por bootstrap dos resíduos do fold de teste
            resids = (y_true - pt["pred"]).values
            rng    = np.random.default_rng(RANDOM_STATE)
            boots  = np.array([fut_series.values +
                               rng.choice(resids, size=len(future_idx),
                                          replace=True)
                               for _ in range(200)])
            plot_future_forecast(
                y, fut_series,
                pd.Series(np.percentile(boots, 2.5,  axis=0), index=future_idx),
                pd.Series(np.percentile(boots, 97.5, axis=0), index=future_idx),
                tid, nome_modelo, OUTPUT_FUTURE
            )
        except _SkipFuture:
            log.info("  Previsão futura desativada (defina RUN_FUTURE=1 para habilitar).")
        except Exception as e:
            log.warning(f"  Future forecast SVR-{mode} failed: {e}")

        resultados.append({
            "transformer":    tid, "label": MAPA_PLOT.get(tid, tid),
            "model":          "SVR", "mode": mode,
            "features_sel":   str(selected_features) if selected_features else "N/A",
            "n_features_sel": len(selected_features) if selected_features else 0,
            "best_params":    str(best_params),
            "MAE_val":        round(best_mae, 4),
            **{k: round(v, 4) for k, v in metrics.items()},
        })

    return pd.DataFrame(resultados)


def main():
    df = carregar_dados()
    df = df[df["id"].isin(TRANSFORMADORES)].copy()
    if ONLY_TID:
        df = df[df["id"] == ONLY_TID].copy()
        log.info(f"ONLY_TID={ONLY_TID} → rodando apenas esse transformador.")

    log.info("=== SVR UNIVARIATE ===")
    res_uv = treinar_svr(df, weather=None, mode="UV")
    res_uv.to_csv(f"{OUTPUT_SVR}/metrics_SVR_UV.csv", index=False)

    log.info("=== SVR MULTIVARIATE ===")
    try:
        weather = carregar_weather()
    except Exception as e:
        log.warning(f"Weather not loaded: {e}"); weather = None
    res_mv = treinar_svr(df, weather=weather, mode="MV")
    res_mv.to_csv(f"{OUTPUT_SVR}/metrics_SVR_MV.csv", index=False)

    res = pd.concat([res_uv, res_mv])
    res.to_csv(f"{OUTPUT_SVR}/metrics_SVR.csv", index=False)
    print(res[["transformer", "model", "mode", "n_features_sel",
               "RMSE", "MAE", "sMAPE", "R2"]]
          .sort_values(["transformer", "RMSE"]).to_string(index=False))


if __name__ == "__main__":
    main()
