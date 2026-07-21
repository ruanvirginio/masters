# =============================================================================
# models_boosting.py — XGBoost · LightGBM · GradientBoosting  (UV and MV)
# =============================================================================
# Methodology (no data leakage):
#
#   TRAIN (2021-2022) → (1) exogenous feature selection
#                     → (2) model parameter fitting
#   VALIDATION (2023) → (3) HPT: select best hyperparameters
#   TEST (2024)       → (4) single final evaluation, reported in the paper
#
# A avaliação do boosting SEMPRE foi recursiva de verdade (backtesting com
# fold único de 365 passos via skforecast). Este rerun serve como sanity
# check: os números devem reproduzir os anteriores.
#
# Feature selection (MV mode):
#   - Run BEFORE HPT, using ONLY training data.
#   - Method: accumulated importance of a fast LightGBM proxy (gain).
#   - Selected features are fixed for all models of the same transformer.
# =============================================================================
import warnings; warnings.filterwarnings("ignore")
import os, random
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from lightgbm  import LGBMRegressor
from xgboost   import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.preprocessing import RollingFeatures

from config import (
    TRANSFORMADORES, MAPA_PLOT, RANDOM_STATE,
    LAGS, ROLLING_WINDOW, END_TRAIN, END_VALIDATION,
    OUTPUT_UV, OUTPUT_MV, OUTPUT_FUTURE, FUTURE_END_DATE
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

# ─────────────────────────────────────────────────────────────────────────────
# CONTROLES DE EXECUÇÃO (rerun rápido) — ver models_dl.py para detalhes.
#   RUN_FUTURE=1 → habilita previsão 2025–2027 (padrão: desligada).
#   ONLY_TID=T1  → roda só um transformador (3 terminais em paralelo).
# ─────────────────────────────────────────────────────────────────────────────
RUN_FUTURE = os.environ.get("RUN_FUTURE", "0") == "1"
ONLY_TID   = os.environ.get("ONLY_TID")

# Cache opcional de best_params (preencha a partir de metrics_UV/MV.csv,
# coluna best_params). Chave: (tid, nome_modelo, mode).
BEST_PARAMS_CACHE = {}


def selecionar_features_mv(y_train: pd.Series,
                            exog_train: pd.DataFrame,
                            tid: str,
                            output_dir: str) -> list:
    """
    Select the most relevant exogenous features using a fast LightGBM proxy
    trained EXCLUSIVELY on training data (no data leakage).
    """
    nome_plot = MAPA_PLOT.get(tid, tid)
    log.info(f"  [FS] Feature selection for {tid} on training set "
             f"({len(y_train)} days, {len(exog_train.columns)} candidates)...")

    proxy = LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        num_leaves=31, random_state=RANDOM_STATE, n_jobs=-1,
        verbose=-1, importance_type="gain",
    )
    fc_proxy = ForecasterRecursive(
        estimator=proxy, lags=LAGS,
        window_features=RollingFeatures(stats=["mean"], window_sizes=[ROLLING_WINDOW])
    )
    fc_proxy.fit(y=y_train, exog=exog_train)
    imp_df   = fc_proxy.get_feature_importances()
    exog_imp = imp_df[imp_df["feature"].isin(exog_train.columns)].copy()
    exog_imp = exog_imp.sort_values("importance", ascending=False).reset_index(drop=True)

    if exog_imp.empty:
        return list(exog_train.columns)

    total_imp = exog_imp["importance"].sum()
    if total_imp == 0:
        return list(exog_train.columns)

    exog_imp["importance_pct"] = exog_imp["importance"] / total_imp
    exog_imp["importance_cum"] = exog_imp["importance_pct"].cumsum()

    selected_mask = exog_imp["importance_cum"].shift(1, fill_value=0) < FEATURE_IMPORTANCE_THRESHOLD
    selected = exog_imp.loc[selected_mask, "feature"].tolist()
    if len(selected) == 0:
        selected = [exog_imp.iloc[0]["feature"]]

    n_total = len(exog_imp)
    n_sel   = len(selected)
    log.info(f"  [FS] {tid}: {n_sel}/{n_total} features selected "
             f"(threshold={FEATURE_IMPORTANCE_THRESHOLD:.0%}): {selected}")

    fig, ax = plt.subplots(figsize=(9, max(4, n_total * 0.5 + 1)))
    colors = ["#2563eb" if f in selected else "#94a3b8" for f in exog_imp["feature"]]
    ax.barh(exog_imp["feature"][::-1], exog_imp["importance"][::-1], color=colors[::-1])
    if n_sel < n_total:
        cutoff_val = exog_imp.iloc[n_sel - 1]["importance"]
        ax.axvline(cutoff_val, color="#dc2626", ls="--", lw=1.2,
                   label=f"Threshold ({FEATURE_IMPORTANCE_THRESHOLD:.0%} accumulated)")
        ax.legend(fontsize=11)
    ax.set_title(f"{nome_plot} — Feature Selection (proxy LightGBM, training set)\n"
                 f"Blue = selected ({n_sel}/{n_total}), Grey = discarded", fontsize=13)
    ax.set_xlabel("Importance (gain)", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/feature_selection_{tid}_MV.svg",
                dpi=150, bbox_inches="tight")
    plt.close()

    exog_imp["selected"] = exog_imp["feature"].isin(selected)
    exog_imp.to_csv(f"{output_dir}/feature_importance_{tid}_proxy.csv", index=False)
    return selected


def get_param_grid():
    return {
        "LGBM": {
            "estimator": LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
            "params": [
                {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 5, "num_leaves": 31},
                {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 6, "num_leaves": 50},
                {"n_estimators": 500, "learning_rate": 0.01, "max_depth": 6, "num_leaves": 50},
                {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4, "num_leaves": 20},
            ]
        },
        "XGBoost": {
            "estimator": XGBRegressor(
                objective="reg:squarederror", random_state=RANDOM_STATE,
                n_jobs=-1, verbosity=0),
            "params": [
                {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8},
                {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 5, "subsample": 0.8},
                {"n_estimators": 500, "learning_rate": 0.01, "max_depth": 4, "subsample": 0.9},
                {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "subsample": 1.0},
            ]
        },
        "GradientBoosting": {
            "estimator": GradientBoostingRegressor(random_state=RANDOM_STATE),
            "params": [
                {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.8},
                {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 4, "subsample": 0.8},
                {"n_estimators": 100, "learning_rate": 0.10, "max_depth": 3, "subsample": 1.0},
            ]
        },
    }


def prever_futuro_bootstrap(forecaster, y_full, exog_full, tid, modelo,
                             output_dir, n_bootstrap=200):
    """Retrain on all data and forecast future period with bootstrap CI."""
    future_idx = pd.date_range(
        y_full.index[-1] + pd.Timedelta(days=1),
        pd.to_datetime(FUTURE_END_DATE), freq="D"
    )
    if len(future_idx) == 0:
        return

    forecaster.fit(y=y_full, exog=exog_full if exog_full is not None else None)

    if exog_full is not None:
        cal_future   = adicionar_features_calendario(pd.Series(np.nan, index=future_idx))
        weather_mean = exog_full[[c for c in exog_full.columns
                                  if c not in cal_future.columns]].mean()
        for col, val in weather_mean.items():
            cal_future[col] = val
        exog_future = cal_future[[c for c in exog_full.columns if c in cal_future.columns]]
        pred_future = forecaster.predict(steps=len(future_idx), exog=exog_future)
    else:
        pred_future = forecaster.predict(steps=len(future_idx))

    pred_series = pd.Series(pred_future.values, index=future_idx)
    resids      = forecaster.in_sample_residuals_["_unknown_level"]
    resids      = resids[~np.isnan(resids)]

    rng   = np.random.default_rng(RANDOM_STATE)
    boots = np.array([
        pred_series.values + rng.choice(resids, size=len(future_idx), replace=True)
        for _ in range(n_bootstrap)
    ])
    lower_95 = np.percentile(boots, 2.5,  axis=0)
    upper_95 = np.percentile(boots, 97.5, axis=0)

    plot_future_forecast(y_full, pred_series,
                         pd.Series(lower_95, index=future_idx),
                         pd.Series(upper_95, index=future_idx),
                         tid, modelo, output_dir)
    pd.DataFrame({
        "date":        future_idx,
        "pred_mean":   pred_series.values,
        "ci_lower_95": lower_95,
        "ci_upper_95": upper_95,
    }).to_csv(f"{output_dir}/future_{tid}_{modelo}.csv", index=False)
    log.info(f"  Future forecast saved: {output_dir}/future_{tid}_{modelo}.csv")


def treinar_boosting(df: pd.DataFrame, weather=None, mode="UV") -> pd.DataFrame:
    """
    mode='UV' → univariate (no exogenous, no feature selection)
    mode='MV' → multivariate (+ calendar + climate, WITH feature selection)
    """
    output_dir = OUTPUT_UV if mode == "UV" else OUTPUT_MV
    param_grid = get_param_grid()
    resultados = []

    for tid, df_tr in df.groupby("id"):
        log.info(f"\n{'='*60}\n  {tid}  [{mode}]\n{'='*60}")
        y      = preparar_serie_diaria(df_tr)
        splits = get_splits(y)

        # ── Seasonal naive baselines (computed once, reported for both UV/MV) ──
        if mode == "UV":
            y_tv_naive = y.loc[splits["train"].union(splits["val"])]
            for period, label in [(7, "Naive-7d"), (365, "Naive-365d")]:
                naive_pred    = seasonal_naive_forecast(y_tv_naive, splits["test"], period=period)
                naive_metrics = compute_metrics(y.loc[splits["test"]], naive_pred)
                log.info(f"  {label} → RMSE={naive_metrics['RMSE']:.4f} R²={naive_metrics['R2']:.4f}")
                resultados.append({
                    "transformer":    tid, "label": MAPA_PLOT.get(tid, tid),
                    "model":          label, "mode": "Naive",
                    "features_sel":   "N/A", "n_features_sel": 0,
                    "best_params":    "N/A", "MAE_val": np.nan,
                    **{k: round(v, 4) for k, v in naive_metrics.items()},
                })

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
                tid, output_dir
            )
            exog = exog_full_df[selected_features]
            log.info(f"  Exogenous features for HPT/train/test: {selected_features}")

        y_tv    = y.loc[splits["train"].union(splits["val"])]
        exog_tv = exog.loc[y_tv.index] if exog is not None else None

        cv_val  = TimeSeriesFold(steps=len(splits["val"]),
                                 initial_train_size=len(splits["train"]), refit=False)
        cv_test = TimeSeriesFold(steps=len(splits["test"]),
                                 initial_train_size=len(y_tv),            refit=False)

        for nome, info in param_grid.items():
            log.info(f"  → {nome}")
            cached = BEST_PARAMS_CACHE.get((tid, nome, mode))
            best_mae, best_params = np.inf, None

            # HPT on validation
            for params in (info["params"] if cached is None else []):
                est = info["estimator"].set_params(**params)
                fc  = ForecasterRecursive(
                    estimator=est, lags=LAGS,
                    window_features=RollingFeatures(stats=["mean"],
                                                    window_sizes=[ROLLING_WINDOW])
                )
                kw = dict(forecaster=fc, y=y_tv, cv=cv_val,
                          metric="mean_absolute_error", verbose=False)
                if exog_tv is not None:
                    kw["exog"] = exog_tv
                _, pv = backtesting_forecaster(**kw)
                mae  = mean_absolute_error(y.loc[pv.index], pv["pred"])
                if mae < best_mae:
                    best_mae, best_params = mae, params

            if cached is not None:
                best_params, best_mae = dict(cached), np.nan
                log.info(f"  ✓ HPT (cache) → {best_params}")
            else:
                log.info(f"  ✓ HPT → {best_params} | MAE_val={best_mae:.4f}")

            # Retrain TRAIN+VAL, evaluate TEST (fold recursivo de 365 passos)
            best_est = info["estimator"].set_params(**best_params)
            fc_final = ForecasterRecursive(
                estimator=best_est, lags=LAGS,
                window_features=RollingFeatures(stats=["mean"],
                                                window_sizes=[ROLLING_WINDOW])
            )
            kw_test = dict(forecaster=fc_final, y=y, cv=cv_test,
                           metric="mean_absolute_error", verbose=False)
            if exog is not None:
                kw_test["exog"] = exog
            _, pt = backtesting_forecaster(**kw_test)

            y_true  = y.loc[pt.index]
            metrics = compute_metrics(y_true, pt["pred"])
            log.info(f"  ✓ TEST → RMSE={metrics['RMSE']:.4f} R²={metrics['R2']:.4f}")

            kw_fit = dict(y=y_tv)
            if exog_tv is not None:
                kw_fit["exog"] = exog_tv
            fc_final.fit(**kw_fit)

            # Validation plot
            fc_val_plot = ForecasterRecursive(
                estimator=info["estimator"].set_params(**best_params), lags=LAGS,
                window_features=RollingFeatures(stats=["mean"],
                                                window_sizes=[ROLLING_WINDOW])
            )
            kw_vp = dict(forecaster=fc_val_plot, y=y_tv, cv=cv_val,
                         metric="mean_absolute_error", verbose=False)
            if exog_tv is not None:
                kw_vp["exog"] = exog_tv
            _, pv_plot = backtesting_forecaster(**kw_vp)

            plot_forecast_3way(y, pv_plot["pred"], pt["pred"],
                               tid, nome, output_dir, suffix=f"_{mode}")
            plot_residuals(y_true, pt["pred"], tid, nome, output_dir)

            pd.DataFrame({
                "date":   pt.index,
                "y_true": y_true.values,
                "y_pred": pt["pred"].values,
            }).to_csv(f"{output_dir}/preds_test_{tid}_{nome}_{mode}.csv", index=False)

            # Feature importance chart
            try:
                imp       = fc_final.get_feature_importances()
                exog_feat = [c for c in imp["feature"]
                             if c in (exog.columns if exog is not None else [])]
                imp_sorted   = imp.sort_values("importance", ascending=True)
                n_show       = min(20, len(imp_sorted))
                imp_top      = imp_sorted.tail(n_show)
                colors_top   = [
                    "#2563eb" if f in exog_feat else "#94a3b8"
                    for f in imp_top["feature"]
                ]
                fig, ax = plt.subplots(figsize=(8, max(4, n_show * 0.42 + 1)))
                ax.barh(imp_top["feature"], imp_top["importance"], color=colors_top)
                ax.set_title(
                    f"{MAPA_PLOT.get(tid,tid)} — {nome} [{mode}]: Feature Importance\n"
                    f"Blue = selected exogenous | Grey = lag / rolling", fontsize=13)
                ax.set_xlabel("Importance (gain)", fontsize=12)
                ax.tick_params(axis="both", labelsize=11)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/importance_{tid}_{nome}_{mode}.svg",
                            dpi=150, bbox_inches="tight")
                plt.close()
            except Exception as e:
                log.warning(f"  Feature importance unavailable: {e}")

            # Future forecast (desligada por padrão no rerun)
            if RUN_FUTURE:
                fc_fut = ForecasterRecursive(
                    estimator=info["estimator"].set_params(**best_params), lags=LAGS,
                    window_features=RollingFeatures(stats=["mean"],
                                                    window_sizes=[ROLLING_WINDOW])
                )
                try:
                    prever_futuro_bootstrap(fc_fut, y, exog, tid, f"{nome}_{mode}",
                                            OUTPUT_FUTURE)
                except Exception as e:
                    log.warning(f"  Future forecast failed ({nome}): {e}")
            else:
                log.info("  Previsão futura desativada (defina RUN_FUTURE=1 para habilitar).")

            resultados.append({
                "transformer":    tid, "label": MAPA_PLOT.get(tid, tid),
                "model":          nome, "mode": mode,
                "features_sel":   str(selected_features) if selected_features else "N/A",
                "n_features_sel": len(selected_features) if selected_features else 0,
                "best_params":    str(best_params),
                "MAE_val":        round(best_mae, 4) if np.isfinite(best_mae) else np.nan,
                **{k: round(v, 4) for k, v in metrics.items()},
            })

    return pd.DataFrame(resultados)


def main():
    df = carregar_dados()
    df = df[df["id"].isin(TRANSFORMADORES)].copy()
    if ONLY_TID:
        df = df[df["id"] == ONLY_TID].copy()
        log.info(f"ONLY_TID={ONLY_TID} → rodando apenas esse transformador.")

    log.info("=== BOOSTING UNIVARIATE ===")
    res_uv = treinar_boosting(df, weather=None, mode="UV")
    res_uv.to_csv(f"{OUTPUT_UV}/metrics_UV.csv", index=False)

    log.info("=== BOOSTING MULTIVARIATE ===")
    try:
        weather = carregar_weather()
    except Exception as e:
        log.warning(f"Weather not loaded: {e}"); weather = None
    res_mv = treinar_boosting(df, weather=weather, mode="MV")
    res_mv.to_csv(f"{OUTPUT_MV}/metrics_MV.csv", index=False)

    print(pd.concat([res_uv, res_mv])
          [["transformer", "model", "mode", "n_features_sel", "RMSE", "MAE", "sMAPE", "R2"]]
          .sort_values(["transformer", "RMSE"]).to_string(index=False))


if __name__ == "__main__":
    main()
