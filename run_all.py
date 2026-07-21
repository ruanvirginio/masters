# =============================================================================
# run_all.py — Orquestrador do rerun completo (protocolo recursivo unificado)
# =============================================================================
# Ordem: SARIMAX → Naive → SVR → Boosting → DL → N-BEATS → stats → Tabela 7.
# Substitui run_all_corrections.py (que não incluía o N-BEATS).
#
# Uso (PowerShell):
#   $env:OMP_NUM_THREADS="12"; python run_all.py            # tudo
#   python run_all.py --skip-dl                             # pula DL+N-BEATS
#   python run_all.py --only-stats                          # só stats+tabela
#   $env:ONLY_TID="T21a"; python run_all.py --skip-stats    # 1 transformador
#     (rode T21a/T22a/T70a em 3 terminais; depois stats SEM ONLY_TID)
#
# RUN_FUTURE=1 no ambiente reativa a previsão 2025-2027 (padrão: desligada).
# =============================================================================
import argparse, os, time
from utils import log


def _run(label, module_name):
    log.info("\n" + "=" * 70)
    log.info(f">>> {label}")
    log.info("=" * 70)
    t0 = time.time()
    mod = __import__(module_name)
    mod.main()
    log.info(f"<<< {label} concluído em {(time.time()-t0)/60:.1f} min")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-dl",    action="store_true",
                   help="pula LSTM/CNN-LSTM e N-BEATS (mais lentos)")
    p.add_argument("--only-stats", action="store_true",
                   help="só stats_tests + build_table7 (assume preds salvos)")
    p.add_argument("--skip-stats", action="store_true",
                   help="só modelos (útil com ONLY_TID em paralelo)")
    args = p.parse_args()

    if os.environ.get("ONLY_TID") and not args.skip_stats and not args.only_stats:
        log.warning("ONLY_TID definido: stats/tabela precisam de TODOS os "
                    "transformadores — rode-os depois, sem ONLY_TID.")

    if not args.only_stats:
        _run("SARIMAX",            "models_sarimax")
        _run("Seasonal Naive",     "models_naive")
        _run("SVR (UV + MV)",      "models_svr")
        _run("Boosting (UV + MV)", "models_boosting")
        if not args.skip_dl:
            _run("LSTM / CNN-LSTM (UV + MV)", "models_dl")
            _run("N-BEATS (UV + MV)",         "models_nbeats")

    if not args.skip_stats:
        _run("Testes estatísticos (DM + bootstrap)", "stats_tests")
        try:
            _run("Tabela 7 (CSV + LaTeX)", "build_table7")
        except Exception as e:
            log.warning(f"build_table7 falhou/ausente: {e}")

    log.info("\n✔ Pipeline concluído.")


if __name__ == "__main__":
    main()
