# =============================================================================
# models_nbeats.py — N-BEATS (Neural Basis Expansion Analysis for Time Series)
# =============================================================================
# N-BEATS: Oreshkin et al. (2020) — ICLR 2020.
#
# Estrutura espelhada ao models_dl.py para comparação simétrica:
#   - HPT roda 1x com DL_SEEDS[0]
#   - Refit roda N_DL_RUNS vezes, uma seed por run
#   - Métricas reportadas = média ± dp sobre os runs
#   - Plots e previsão futura usam o melhor run
#
# Modo UV:  entrada = janela (lookback, 1) — só Smax normalizado
# Modo MV:  entrada = janela (lookback, 1 + n_exog_sel)
#
# >>> CORREÇÃO (avaliação recursiva no TESTE) <<<
# A versão anterior avaliava o teste com model.predict(X_te), cujas janelas
# contêm os VALORES REAIS de 2024 (teacher forcing / 1 passo à frente).
# Agora o teste usa recursive_predict_nbeats: janela inicial = últimos 365
# dias reais (termina em 31/12/2023); a cada passo, a previsão realimenta a
# janela. Exógenas do teste (calendário/climatologia) são determinísticas e
# entram com seus valores reais escalados — não é vazamento.
#
# Scalers fitados APENAS no treino → sem data leakage.
# HPT: treina em 2021-2022 (early stopping em val_loss) e SELECIONA pelo MAE
# da previsão RECURSIVA de 365 passos sobre 2023 — Fold 1 simétrico ao
# boosting. Refit monitora "loss" (X_vl ⊂ X_tv, val_loss vazaria).
# =============================================================================
import warnings; warnings.filterwarnings("ignore")
import os, random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
try:
    import keras
    from keras.models import Model
    from keras.layers import Input, Dense, Subtract, Add, Flatten, Lambda
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.optimizers import Adam
    import keras.backend as K
except ImportError:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Subtract, Add, Flatten, Lambda
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    import tensorflow.keras.backend as K

from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor

from config import (
    DATA_PATH, TRANSFORMADORES, MAPA_PLOT, RANDOM_STATE,
    DL_LOOKBACK, DL_EPOCHS, DL_BATCH_SIZE, DL_PATIENCE, DL_LR,
    END_TRAIN, END_VALIDATION, OUTPUT_DL, OUTPUT_FUTURE, FUTURE_END_DATE,
    N_DL_RUNS, DL_SEEDS
)
from utils import (
    preparar_serie_diaria, adicionar_features_calendario,
    carregar_weather, compute_metrics, get_splits,
    plot_forecast_3way, plot_residuals, plot_future_forecast, log
)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


EXOG_COLS = [
    "weekday_sin", "weekday_cos", "month_sin", "month_cos",
    "dayofyear_sin", "dayofyear_cos", "is_weekend", "is_holiday",
    "temp_max_hist", "temp_min_hist", "temp_mean_hist", "precip_hist"
]
FEATURE_IMPORTANCE_THRESHOLD = 0.95

OUTPUT_NBEATS = OUTPUT_DL

# ─────────────────────────────────────────────────────────────────────────────
# CONTROLES DE EXECUÇÃO (rerun rápido) — ver models_dl.py para detalhes.
# ─────────────────────────────────────────────────────────────────────────────
RUN_FUTURE = os.environ.get("RUN_FUTURE", "0") == "1"
ONLY_TID   = os.environ.get("ONLY_TID")

class _SkipFuture(Exception):
    pass

# Cache de hiperparâmetros — VAZIO de propósito: o critério da HPT mudou de
# MAE de 1 passo para MAE RECURSIVO de 365 passos na validação (Fold 1,
# simétrico ao boosting). Preencha apenas com resultados desta versão.
# Obs.: os IDs reais são os do CSV (ex.: T21a), não os anonimizados (T1).
BEST_PARAMS_CACHE = {}


# ─────────────────────────────────────────────────────────────────────────────
# N-BEATS BLOCK
# ─────────────────────────────────────────────────────────────────────────────
def nbeats_block(x, input_size, theta_size, hidden_units, n_layers):
    h = x
    for _ in range(n_layers):
        h = Dense(hidden_units, activation="relu")(h)

    theta = Dense(theta_size, activation="linear", use_bias=False)(h)

    theta_b = Lambda(lambda t: t[:, :input_size])(theta)
    theta_f = Lambda(lambda t: t[:, input_size:])(theta)

    backcast = Dense(input_size, activation="linear", use_bias=False)(theta_b)
    forecast = Dense(1,          activation="linear", use_bias=False)(theta_f)

    return backcast, forecast


# ─────────────────────────────────────────────────────────────────────────────
# N-BEATS MODEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_nbeats(n_feat, n_stacks=2, n_blocks_per_stack=3,
                 hidden_units=256, n_layers=4, theta_size=None):
    lookback   = DL_LOOKBACK
    input_size = lookback * n_feat
    if theta_size is None:
        theta_size = input_size + 1

    x_in   = Input(shape=(lookback, n_feat), name="input")
    x_flat = Flatten()(x_in)

    residual  = x_flat
    forecasts = []

    for stack_id in range(n_stacks):
        for block_id in range(n_blocks_per_stack):
            backcast, forecast = nbeats_block(
                residual, input_size, theta_size, hidden_units, n_layers
            )
            residual = Subtract(name=f"residual_s{stack_id}_b{block_id}")(
                [residual, backcast]
            )
            forecasts.append(forecast)

    if len(forecasts) == 1:
        total_forecast = forecasts[0]
    else:
        total_forecast = Add(name="global_forecast")(forecasts)

    model = Model(inputs=x_in, outputs=total_forecast, name="NBEATS")
    model.compile(Adam(DL_LR), "mse", metrics=["mae"])
    return model


ARCHITECTURES = {
    "NBEATS": (build_nbeats, [
        {"n_stacks": 2, "n_blocks_per_stack": 3, "hidden_units": 128, "n_layers": 4},
        {"n_stacks": 2, "n_blocks_per_stack": 3, "hidden_units": 256, "n_layers": 4},
        {"n_stacks": 3, "n_blocks_per_stack": 3, "hidden_units": 256, "n_layers": 4},
    ]),
}


# ─────────────────────────────────────────────────────────────────────────────
# SELEÇÃO DE FEATURES (idêntica ao boosting / DL)
# ─────────────────────────────────────────────────────────────────────────────
def selecionar_features_mv(y_train, exog_train, tid, output_dir):
    proxy = LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        num_leaves=31, random_state=RANDOM_STATE, n_jobs=-1,
        verbose=-1, importance_type="gain"
    )
    from skforecast.recursive import ForecasterRecursive
    from skforecast.preprocessing import RollingFeatures
    from config import LAGS, ROLLING_WINDOW
    fc = ForecasterRecursive(
        estimator=proxy, lags=LAGS,
        window_features=RollingFeatures(stats=["mean"], window_sizes=[ROLLING_WINDOW])
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

    log.info(f"  [FS-NBEATS] {tid}: {len(selected)}/{len(exog_imp)} → {selected}")

    colors = ["#2563eb" if f in selected else "#94a3b8" for f in exog_imp["feature"]]
    fig, ax = plt.subplots(figsize=(9, max(4, len(exog_imp) * 0.5 + 1)))
    ax.barh(exog_imp["feature"][::-1], exog_imp["importance"][::-1], color=colors[::-1])
    ax.set_title(f"{MAPA_PLOT.get(tid,tid)} — N-BEATS: Feature Selection\n"
                 f"Blue = selected ({len(selected)}/{len(exog_imp)})", fontsize=13)
    ax.set_xlabel("Importance (gain)", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/feature_selection_{tid}_NBEATS_MV.svg",
                dpi=150, bbox_inches="tight")
    plt.close()
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# PREPARAÇÃO DE DADOS (idêntica ao models_dl.py)
# ─────────────────────────────────────────────────────────────────────────────
def preparar_dados_nbeats(df_tr, weather=None, mode="UV"):
    y = preparar_serie_diaria(df_tr)
    splits = get_splits(y)

    sel_features = None

    if mode == "MV":
        cal = adicionar_features_calendario(y)
        if weather is not None:
            exog_df = cal.join(weather, how="left").bfill().ffill()
        else:
            exog_df = cal
        cols_disp = [c for c in EXOG_COLS if c in exog_df.columns]
        exog_df = exog_df[cols_disp]

        sel_features = selecionar_features_mv(
            y.loc[splits["train"]],
            exog_df.loc[splits["train"]],
            df_tr["id"].iloc[0] if "id" in df_tr.columns else "?",
            OUTPUT_NBEATS
        )
        exog_sel = exog_df[[c for c in sel_features if c in exog_df.columns]]

        sc_exog = MinMaxScaler()
        sc_exog.fit(exog_sel.loc[splits["train"]].values)
        exog_sc = sc_exog.transform(exog_sel.values)
    else:
        exog_sc      = None
        sc_exog      = None
        sel_features = []

    y_raw = y.values.reshape(-1, 1).astype(np.float32)

    train_vals = y_raw[y.index.isin(splits["train"])]
    if np.isnan(train_vals).any():
        median_train = float(np.nanmedian(train_vals))
        log.warning(f"  NaN detectado em y_train após interpolação — preenchendo com mediana ({median_train:.4f})")
        y_raw = np.where(np.isnan(y_raw), median_train, y_raw)

    sc_y = MinMaxScaler()
    sc_y.fit(y_raw[y.index.isin(splits["train"])])
    y_sc = sc_y.transform(y_raw).flatten().astype(np.float32)
    y_sc = np.nan_to_num(y_sc, nan=0.5, posinf=1.0, neginf=0.0)

    if exog_sc is not None:
        X_raw = np.hstack([y_sc.reshape(-1, 1), exog_sc]).astype(np.float32)
    else:
        X_raw = y_sc.reshape(-1, 1).astype(np.float32)
    X_raw = np.nan_to_num(X_raw, nan=0.5, posinf=1.0, neginf=0.0)

    def make_windows(X, y, lb):
        Xw, yw = [], []
        for i in range(lb, len(y)):
            Xw.append(X[i - lb:i])
            yw.append(y[i])
        return np.array(Xw, dtype=np.float32), np.array(yw, dtype=np.float32)

    Xw, yw = make_windows(X_raw, y_sc, DL_LOOKBACK)
    idx_w  = y.index[DL_LOOKBACK:]

    t_end = pd.to_datetime(END_TRAIN)
    v_end = pd.to_datetime(END_VALIDATION)
    m_tr  = idx_w <= t_end
    m_vl  = (idx_w > t_end) & (idx_w <= v_end)
    m_te  = idx_w > v_end

    return dict(
        X_tr=Xw[m_tr], y_tr=yw[m_tr],
        X_vl=Xw[m_vl], y_vl=yw[m_vl],
        X_te=Xw[m_te], y_te=yw[m_te],
        idx_vl=idx_w[m_vl], idx_te=idx_w[m_te],
        sc_y=sc_y, sc_exog=sc_exog,
        y_series=y, n_feat=Xw.shape[2],
        X_all=Xw, y_all=yw, idx_all=idx_w,
        sel_features=sel_features,
        X_raw_full=X_raw,
        y_sc_full=y_sc,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────
def get_callbacks_hpt():
    """HPT: X_vl NÃO está em X_tr → monitorar val_loss é seguro."""
    return [
        EarlyStopping(monitor="val_loss", patience=DL_PATIENCE,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=DL_PATIENCE // 2, min_lr=1e-6, verbose=0),
    ]


def get_callbacks_refit():
    """Refit em TRAIN+VAL: X_vl ⊂ X_tv → monitora a perda de treino."""
    return [
        EarlyStopping(monitor="loss", patience=DL_PATIENCE,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="loss", factor=0.5,
                          patience=DL_PATIENCE // 2, min_lr=1e-6, verbose=0),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# RECURSIVE PREDICT
# ─────────────────────────────────────────────────────────────────────────────
def recursive_predict_nbeats(model, last_window, fut_exog_sc, sc_y, steps):
    """
    Previsão recursiva multi-passos (Paper, Seção 3.5 / Eq. 7).
    UV: fut_exog_sc = None → desliza janela só com previsão anterior.
    MV: fut_exog_sc = (steps, n_exog) → canal exógeno determinístico por passo.

    Usa model(x, training=False) em vez de .predict() para eliminar o
    overhead de grafo em `steps` chamadas sequenciais (≈10× mais rápido).
    """
    window   = last_window.copy()   # (lookback, n_feat)
    preds_sc = []
    n_feat   = window.shape[1]

    # Compila a chamada uma única vez (elimina retraçamento por passo).
    pred_fn = tf.function(
        lambda x: model(x, training=False),
        input_signature=[tf.TensorSpec([1, window.shape[0], n_feat], tf.float32)],
    )

    for s in range(steps):
        x_in = tf.constant(window[np.newaxis].astype(np.float32))
        yhat = float(pred_fn(x_in).numpy()[0, 0])
        preds_sc.append(yhat)

        new_row = np.zeros(n_feat, dtype=np.float32)
        new_row[0] = yhat
        if fut_exog_sc is not None and s < len(fut_exog_sc):
            new_row[1:] = fut_exog_sc[s]
        window = np.vstack([window[1:], new_row])

    preds_sc = np.array(preds_sc).reshape(-1, 1)
    return sc_y.inverse_transform(preds_sc).flatten()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def treinar_nbeats(df: pd.DataFrame, weather=None, mode="UV") -> pd.DataFrame:
    resultados = []

    for tid, df_tr in df.groupby("id"):
        log.info(f"\n{'='*60}\n  {tid}  [NBEATS-{mode}]\n{'='*60}")
        data = preparar_dados_nbeats(df_tr, weather=weather, mode=mode)

        if len(data["X_tr"]) == 0 or len(data["X_vl"]) == 0:
            log.warning(f"  {tid}: dados insuficientes."); continue

        for key in ["X_tr", "y_tr", "X_vl", "y_vl", "X_te", "y_te", "X_all", "y_all"]:
            data[key] = np.array(data[key], dtype=np.float32)

        n_test = len(data["idx_te"])

        # ── Insumos da RECURSÃO DE TESTE (idênticos para todos os runs) ──
        # X_te[0] = janela dos 365 dias REAIS que termina em 31/12/2023.
        window_inicial_te = data["X_te"][0].copy()
        # Exógenas dos dias de teste (MV): determinísticas, já escaladas.
        exog_te_sc = None
        if mode == "MV" and data["n_feat"] > 1:
            exog_te_sc = data["X_raw_full"][-n_test:, 1:]

        # ── Insumos da RECURSÃO DE VALIDAÇÃO (HPT / Fold 1) ──────────────
        n_vl = len(data["idx_vl"])
        window_inicial_vl = data["X_vl"][0].copy()
        exog_vl_sc = None
        if mode == "MV" and data["n_feat"] > 1:
            exog_vl_sc = data["X_raw_full"][-(n_vl + n_test):-n_test, 1:]
        yv_inv = data["sc_y"].inverse_transform(
                     data["y_vl"].reshape(-1, 1)).flatten()

        yt_inv   = data["sc_y"].inverse_transform(
                       data["y_te"].reshape(-1, 1)).flatten()
        y_true_s = pd.Series(yt_inv, index=data["idx_te"])

        for nome, (build_fn, hpt_grid) in ARCHITECTURES.items():
            log.info(f"\n  >>> {nome} [{mode}]")

            # ── HPT sobre VALIDAÇÃO ──────────────────────────────────────
            set_all_seeds(DL_SEEDS[0])
            cached = BEST_PARAMS_CACHE.get((tid, nome, mode))
            best_mae, best_params = np.inf, None
            for params in (hpt_grid if cached is None else []):
                try:
                    model = build_fn(data["n_feat"], **params)
                    model.fit(data["X_tr"], data["y_tr"],
                              validation_data=(data["X_vl"], data["y_vl"]),
                              epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE,
                              callbacks=get_callbacks_hpt(), verbose=0)
                    # Critério = tarefa: MAE da previsão RECURSIVA de 365
                    # passos sobre a validação (Fold 1).
                    pv_inv = recursive_predict_nbeats(
                        model, window_inicial_vl, exog_vl_sc,
                        data["sc_y"], n_vl)
                    mae = float(np.mean(np.abs(yv_inv - pv_inv)))
                    if not np.isfinite(mae):
                        log.warning(f"     {params} → MAE_val=nan/inf (skipped)")
                        continue
                    log.info(f"     {params} → MAE_val={mae:.4f}")
                    if mae < best_mae:
                        best_mae, best_params = mae, params
                except Exception as e:
                    log.warning(f"     {params} → falhou: {e}")
                    continue

            if cached is not None:
                best_params, best_mae = dict(cached), np.nan
                log.info(f"  ✓ HPT (cache) → {best_params}")

            if best_params is None:
                log.error(f"  {tid}/{nome}: nenhuma configuração HPT convergiu. Pulando.")
                continue

            log.info(f"  ✓ HPT → {best_params} | MAE_val={best_mae:.4f}")

            # ─────────────────────────────────────────────────────────────
            # N_DL_RUNS runs com sementes distintas.
            # AVALIAÇÃO DE TESTE RECURSIVA (Paper Seção 3.5): NÃO se usa
            # model.predict(X_te) — as janelas de X_te contêm os valores
            # REAIS de 2024 e produziriam previsão de 1 passo (teacher
            # forcing), incompatível com o protocolo recursivo.
            # ─────────────────────────────────────────────────────────────
            X_tv = np.vstack([data["X_tr"], data["X_vl"]])
            y_tv = np.concatenate([data["y_tr"], data["y_vl"]])

            run_metrics   = []
            run_preds     = []
            best_run_idx  = None
            best_run_rmse = np.inf
            best_history  = None
            best_model    = None
            best_pred_te  = None
            best_y_true   = None

            for run_id, seed in enumerate(DL_SEEDS):
                log.info(f"     [run {run_id+1}/{N_DL_RUNS}] seed={seed}")
                set_all_seeds(seed)

                model_run = build_fn(data["n_feat"], **best_params)
                history = model_run.fit(
                    X_tv, y_tv,
                    callbacks=get_callbacks_refit(),
                    epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE, verbose=0
                )

                # >>> PREVISÃO RECURSIVA DE 365 PASSOS (teste 2024) <<<
                pt_inv = recursive_predict_nbeats(
                    model_run, window_inicial_te, exog_te_sc,
                    data["sc_y"], n_test
                )
                y_pred_s = pd.Series(pt_inv, index=data["idx_te"])
                m = compute_metrics(y_true_s, y_pred_s)
                m["seed"] = seed
                run_metrics.append(m)
                run_preds.append(pt_inv)

                if m["RMSE"] < best_run_rmse:
                    best_run_rmse = m["RMSE"]
                    best_run_idx  = run_id
                    best_history  = history
                    best_model    = model_run
                    best_pred_te  = y_pred_s.copy()
                    best_y_true   = y_true_s.copy()

                pd.DataFrame({
                    "date":   data["idx_te"],
                    "y_true": yt_inv,
                    "y_pred": pt_inv,
                    "seed":   seed,
                }).to_csv(
                    f"{OUTPUT_NBEATS}/preds_test_{tid}_{nome}_{mode}_seed{seed}.csv",
                    index=False
                )

            # Agregar métricas: média ± dp ao longo dos runs
            df_runs = pd.DataFrame(run_metrics)
            agg = {
                "RMSE_mean":  float(df_runs["RMSE"].mean()),
                "RMSE_std":   float(df_runs["RMSE"].std(ddof=1)),
                "MAE_mean":   float(df_runs["MAE"].mean()),
                "MAE_std":    float(df_runs["MAE"].std(ddof=1)),
                "sMAPE_mean": float(df_runs["sMAPE"].mean()),
                "sMAPE_std":  float(df_runs["sMAPE"].std(ddof=1)),
                "R2_mean":    float(df_runs["R2"].mean()),
                "R2_std":     float(df_runs["R2"].std(ddof=1)),
            }
            log.info(f"  ✓ TESTE (média ± dp de {N_DL_RUNS} runs) → "
                     f"RMSE={agg['RMSE_mean']:.4f}±{agg['RMSE_std']:.4f}  "
                     f"R²={agg['R2_mean']:.4f}±{agg['R2_std']:.4f}")

            df_runs.to_csv(
                f"{OUTPUT_NBEATS}/runs_detail_{tid}_{nome}_{mode}.csv", index=False
            )

            # Ensemble (média dos runs) — usado para DM e bootstrap
            ensemble_pred = np.mean(np.vstack(run_preds), axis=0)
            pd.DataFrame({
                "date":   data["idx_te"],
                "y_true": yt_inv,
                "y_pred": ensemble_pred,
            }).to_csv(
                f"{OUTPUT_NBEATS}/preds_test_{tid}_{nome}_{mode}.csv",
                index=False
            )

            # ── Plots: usam o melhor run ─────────────────────────────────
            model_final = best_model
            history     = best_history
            y_pred_s    = best_pred_te

            # Previsão de validação para o plot (1 passo — ilustrativa)
            pv2_sc  = model_final.predict(data["X_vl"], verbose=0).flatten()
            pv2_inv = data["sc_y"].inverse_transform(pv2_sc.reshape(-1, 1)).flatten()
            pred_vl_s = pd.Series(pv2_inv, index=data["idx_vl"])

            nome_modelo = f"{nome}_{mode}"
            plot_forecast_3way(data["y_series"], pred_vl_s, y_pred_s,
                               tid, nome_modelo, OUTPUT_NBEATS, suffix="")
            plot_residuals(best_y_true, y_pred_s, tid, nome_modelo, OUTPUT_NBEATS)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(history.history["loss"], label="Training loss",
                    color="#2563eb", lw=1.5)
            ax.set_title(f"{MAPA_PLOT.get(tid,tid)} — {nome_modelo}: Training Loss (MSE)\n"
                         f"(best of {N_DL_RUNS} runs, seed={DL_SEEDS[best_run_idx]})",
                         fontsize=13)
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("MSE Loss", fontsize=12)
            ax.tick_params(axis="both", labelsize=11)
            ax.legend(fontsize=11)
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_NBEATS}/loss_{tid}_{nome_modelo}.svg",
                        dpi=150, bbox_inches="tight"); plt.close()

            # ── Previsão futura (apenas 1 run, com seed principal) ───────
            try:
                if not RUN_FUTURE:
                    raise _SkipFuture
                set_all_seeds(DL_SEEDS[0])
                future_idx = pd.date_range(
                    data["y_series"].index[-1] + pd.Timedelta(days=1),
                    pd.to_datetime(FUTURE_END_DATE), freq="D"
                )
                fut_exog_sc = None
                if mode == "MV" and data["sc_exog"] is not None:
                    cal_fut = adicionar_features_calendario(
                        pd.Series(np.nan, index=future_idx))
                    if weather is not None:
                        for col in [c for c in data["sel_features"]
                                    if c not in cal_fut.columns and c in weather.columns]:
                            cal_fut[col] = weather[col].mean()
                    sel_fut = [c for c in data["sel_features"] if c in cal_fut.columns]
                    fut_raw = cal_fut[sel_fut].values.astype(np.float32)
                    n_exog  = data["n_feat"] - 1
                    if fut_raw.shape[1] < n_exog:
                        pad = np.zeros((len(future_idx), n_exog - fut_raw.shape[1]),
                                       dtype=np.float32)
                        fut_raw = np.hstack([fut_raw, pad])
                    fut_exog_sc = data["sc_exog"].transform(fut_raw[:, :n_exog])

                model_fut = build_fn(data["n_feat"], **best_params)
                model_fut.fit(data["X_all"], data["y_all"],
                              epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE,
                              callbacks=get_callbacks_refit(), verbose=0)

                preds_fut = recursive_predict_nbeats(
                    model_fut, data["X_all"][-1], fut_exog_sc,
                    data["sc_y"], len(future_idx)
                )
                fut_series = pd.Series(preds_fut, index=future_idx)

                all_pred_inv = data["sc_y"].inverse_transform(
                    model_fut.predict(data["X_all"], verbose=0).reshape(-1, 1)).flatten()
                all_y_inv = data["sc_y"].inverse_transform(
                    data["y_all"].reshape(-1, 1)).flatten()
                resids = all_y_inv - all_pred_inv
                rng    = np.random.default_rng(RANDOM_STATE)
                boots  = np.array([preds_fut + rng.choice(resids, size=len(future_idx),
                                                           replace=True)
                                   for _ in range(200)])
                plot_future_forecast(
                    data["y_series"], fut_series,
                    pd.Series(np.percentile(boots, 2.5,  axis=0), index=future_idx),
                    pd.Series(np.percentile(boots, 97.5, axis=0), index=future_idx),
                    tid, nome_modelo, OUTPUT_FUTURE
                )
            except _SkipFuture:
                log.info("  Previsão futura desativada (defina RUN_FUTURE=1 para habilitar).")
            except Exception as e:
                log.warning(f"  Previsão futura {nome_modelo} falhou: {e}")

            resultados.append({
                "transformador": tid, "nome_plot": MAPA_PLOT.get(tid, tid),
                "modelo": nome, "abordagem": mode,
                "features_sel":   str(data["sel_features"]),
                "n_features_sel": len(data["sel_features"]),
                "best_params":    str(best_params),
                "MAE_val":        round(best_mae, 4),
                "n_runs":         N_DL_RUNS,
                "RMSE":  round(agg["RMSE_mean"],  4),
                "MAE":   round(agg["MAE_mean"],   4),
                "sMAPE": round(agg["sMAPE_mean"], 4),
                "R2":    round(agg["R2_mean"],    4),
                **{k: round(v, 4) for k, v in agg.items()},
            })

    return pd.DataFrame(resultados)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(DATA_PATH, sep=";", encoding="latin-1")
    df = df[df["id"].isin(TRANSFORMADORES)].copy()
    if ONLY_TID:
        df = df[df["id"] == ONLY_TID].copy()
        log.info(f"ONLY_TID={ONLY_TID} → rodando apenas esse transformador.")

    weather = None
    try:
        weather = carregar_weather()
    except Exception as e:
        log.warning(f"Clima: {e}")

    log.info("=== N-BEATS UNIVARIADO ===")
    res_uv = treinar_nbeats(df, weather=None, mode="UV")
    res_uv.to_csv(f"{OUTPUT_NBEATS}/metricas_NBEATS_UV.csv", index=False)

    log.info("=== N-BEATS MULTIVARIADO ===")
    res_mv = treinar_nbeats(df, weather=weather, mode="MV")
    res_mv.to_csv(f"{OUTPUT_NBEATS}/metricas_NBEATS_MV.csv", index=False)

    res = pd.concat([res_uv, res_mv])
    res.to_csv(f"{OUTPUT_NBEATS}/metricas_NBEATS.csv", index=False)
    print(res[["transformador", "modelo", "abordagem", "RMSE", "MAE", "sMAPE", "R2"]]
          .sort_values(["transformador", "abordagem"]).to_string(index=False))


if __name__ == "__main__":
    main()
