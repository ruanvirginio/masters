# Previsão de Demanda Elétrica em Transformadores de Distribuição Urbana

**Um Estudo Comparativo de Previsão de Médio Prazo com Aprendizado de Máquina**

Repositório de código, dados e experimentos da dissertação de mestrado de **Ruan Carlos Virginio dos Santos**, apresentada ao Programa de Pós-Graduação em Tecnologia da Informação (PPGTI) do **Instituto Federal da Paraíba (IFPB)**, João Pessoa — PB, 2026.

- **Orientador:** Prof. Dr. Diego Ernesto Rosa Pessoa
- **Coorientador:** Prof. Dr. Thiago José Marques Moura

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![status](https://img.shields.io/badge/status-reproducible-brightgreen)

---

## Sumário

- [Visão geral](#visão-geral)
- [Questões de pesquisa](#questões-de-pesquisa)
- [Modelos avaliados](#modelos-avaliados)
- [Protocolo experimental](#protocolo-experimental)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Requisitos](#requisitos)
- [Como executar](#como-executar)
- [Dados](#dados)
- [Saídas produzidas](#saídas-produzidas)
- [Reprodutibilidade](#reprodutibilidade)
- [Principais achados](#principais-achados)
- [Limitações](#limitações)
- [Como citar](#como-citar)
- [Licença](#licença)

---

## Visão geral

Este trabalho avalia, sob **protocolo experimental controlado e livre de vazamento**, o desempenho de oito modelos de aprendizado de máquina na **previsão recursiva de demanda elétrica diária máxima ($S_{\max}$)** em três transformadores de força de subestações de distribuição em João Pessoa (PB), com **horizonte de 365 dias** (2024 completo).

A contribuição central é caracterizar como a interação entre **arquitetura algorítmica** e **atributos exógenos** (calendário e climatologia) varia conforme o **perfil de demanda** do ativo (comercial, residencial e misto costeiro), fornecendo diretrizes objetivas de seleção de modelo por transformador.

## Questões de pesquisa

1. Em um horizonte recursivo de 365 dias no nível de transformadores de distribuição, dados históricos de carga são suficientes para previsão acurada, ou a suficiência depende criticamente da arquitetura empregada?
2. O ganho preditivo proveniente de atributos exógenos (calendários e climatológicos) é universal entre algoritmos, ou varia em função da arquitetura e da composição local da demanda?
3. O melhor modelo é o mesmo para transformadores com perfis distintos, ou a escolha adequada depende de características identificáveis previamente em cada ativo?

## Modelos avaliados

Oito modelos, distribuídos em seis famílias algorítmicas, avaliados nas configurações **univariada (UV)** e **multivariada (MV)**:

| Família | Modelo | Script |
|---|---|---|
| Baseline | Seasonal Naive (s=7, s=365) | `models_naive.py` |
| Estatística | SARIMAX | `models_sarimax.py` |
| Kernel | SVR (RBF) | `models_svr.py` |
| Boosting | LightGBM, XGBoost, Gradient Boosting | `models_boosting.py` |
| Aprendizado profundo | LSTM, CNN-LSTM | `models_dl.py` |
| Expansão de base | N-BEATS | `models_nbeats.py` |

## Protocolo experimental

- **Partição temporal cronológica:** Treino 2021–2022 · Validação 2023 · Teste 2024.
- **Dois folds livre de vazamento:**
  - **Fold 1 (seleção):** ajuste no treino, HPT por **MAE recursivo de 365 passos** na validação.
  - **Fold 2 (avaliação):** retreino em 2021–2023 e **previsão recursiva única** de 366 passos sobre 2024.
- **Seleção de atributos MV:** importância acumulada de ganho (proxy LightGBM, $\tau = 95\%$), calculada **somente sobre o treino**.
- **Testes estatísticos:**
  - **Diebold–Mariano** com correção **HLN** ($\alpha = 0{,}05$) contra o melhor modelo de cada transformador.
  - **Block bootstrap** (bloco = 30 dias, 1.000 reamostragens) para IC 95% das métricas.
- **Aprendizado profundo:** 5 sementes ($\{42, 123, 456, 789, 2024\}$); média ± desvio-padrão reportados.
- **Métricas:** RMSE, MAE, sMAPE (primária de reporte) e R².

## Estrutura do repositório

```
.
├── config.py                 # Configuração central (paths, splits, hiperparâmetros)
├── utils.py                  # Utilitários compartilhados (métricas, splits, DM, bootstrap)
├── run_all.py                # Orquestrador do pipeline completo
│
├── EDA_disser.py             # Análise exploratória (Capítulo 4)
│
├── models_naive.py           # Baseline sazonal
├── models_sarimax.py         # SARIMAX
├── models_svr.py             # SVR (UV + MV)
├── models_boosting.py        # LightGBM, XGBoost, GradientBoosting (UV + MV)
├── models_dl.py              # LSTM, CNN-LSTM (UV + MV)
├── models_nbeats.py          # N-BEATS (UV + MV)
│
├── stats_tests.py            # Diebold-Mariano + block bootstrap
├── build_table7.py           # Consolida a Tabela 7 (CSV + LaTeX)
│
├── bases_tratadas/           # Dados de entrada (não versionados)
│   ├── daily_peak_transformers_dataset.csv
│   └── weather_dataset_2022-2024.csv
│
└── resultados_*/             # Saídas por modelo (criadas na execução)
```

## Requisitos

- Python **3.10+**
- Bibliotecas principais:
  - `numpy`, `pandas`, `scipy`, `scikit-learn`
  - `statsmodels`, `holidays`
  - `lightgbm`, `xgboost`
  - `skforecast` (backtesting recursivo)
  - `tensorflow` (LSTM/CNN-LSTM)
  - `matplotlib` (renderização em backend `Agg`)

Instalação sugerida (ambiente virtual):

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows PowerShell
pip install -r requirements.txt
```

## Como executar

O ponto de entrada é `run_all.py`. A ordem interna é: SARIMAX → Seasonal Naive → SVR → Boosting → DL (LSTM/CNN-LSTM) → N-BEATS → testes estatísticos → Tabela 7.

**Execução completa (Linux/macOS):**
```bash
export OMP_NUM_THREADS=12
python run_all.py
```

**Execução completa (Windows PowerShell):**
```powershell
$env:OMP_NUM_THREADS="12"
python run_all.py
```

**Modos parciais úteis:**
```bash
python run_all.py --skip-dl        # pula LSTM/CNN-LSTM e N-BEATS
python run_all.py --only-stats     # só stats + Tabela 7 (assume preds salvos)
python run_all.py --skip-stats     # só modelos
```

**Paralelização por transformador** (três terminais, um por ativo):
```bash
ONLY_TID=T21a python run_all.py --skip-stats
ONLY_TID=T22a python run_all.py --skip-stats
ONLY_TID=T70a python run_all.py --skip-stats
# ao final, sem ONLY_TID:
python run_all.py --only-stats
```

**Análise exploratória (independente do pipeline):**
```bash
python EDA_disser.py
```

## Dados

O conjunto de dados combina duas fontes:

1. **Medições SCADA** de três transformadores de força de subestações de distribuição em João Pessoa, no período de **1º/01/2021 a 31/12/2024** (1.461 dias, granularidade horária), das quais se deriva a potência aparente máxima diária ($S_{\max}$). Identificadores originais foram substituídos por códigos anonimizados:

   | Interno | Anonimizado | Perfil |
   |:---:|:---:|:---|
   | T21a | **T1** | Comercial (centro urbano) |
   | T22a | **T2** | Residencial (classe média/média-baixa) |
   | T70a | **T3** | Misto costeiro (turismo, alta temporada) |

2. **Dados meteorológicos** da estação automática do **INMET** em João Pessoa (2022–2024), a partir dos quais são calculadas **normais climatológicas diárias** (temperatura máx./mín./média e precipitação acumulada) por dia do ano, excluindo 2024 do cálculo para preservar a integridade do teste.

Arquivos esperados em `bases_tratadas/`:
- `daily_peak_transformers_dataset.csv` (medições diárias já regularizadas)
- `weather_dataset_2022-2024.csv` (normais climatológicas)

Datasets processados serão disponibilizados junto ao repositório para reprodução; o mapeamento entre códigos anonimizados e ativos reais é mantido privado por acordo com a concessionária.

## Saídas produzidas

Ao final da execução, os diretórios abaixo contêm os artefatos utilizados nas tabelas e figuras do Capítulo 5:

```
resultados_SARIMAX/     metricas_SARIMAX.csv · preds_test_*.csv · figuras
resultados_NAIVE/       metricas_SeasonalNaive.csv · preds_test_*.csv
resultados_SVR/         metricas_SVR.csv · preds_test_*.csv · figuras
resultados_UV/          metrics_UV.csv (boosting UV) · preds_test_*.csv
resultados_MV/          metrics_MV.csv (boosting MV) · preds_test_*.csv
resultados_DL/          metricas_DL.csv · runs_detail_*.csv (5 sementes) · preds_test_*.csv
resultados_stats/       diebold_mariano_pairs.csv · bootstrap_ci_test_metrics.csv
relatorio/              table7_final.csv · table7_final.tex
eda_out/                figuras e tabelas do Capítulo 4
```

## Reprodutibilidade

O projeto foi desenhado para reprodução completa a partir do commit versionado:

- **Semente global:** `RANDOM_STATE = 42` (`config.py`), propagada a `numpy`, `random`, `tensorflow` e `PYTHONHASHSEED`.
- **Aprendizado profundo:** cinco execuções independentes com sementes `[42, 123, 456, 789, 2024]`, cada uma com `set_all_seeds(seed)` chamado antes do `fit`.
- **Sem vazamento temporal:** normalização Min-Max, climatologias, seleção de atributos e HPT usam **apenas** dados anteriores ao conjunto avaliado. Recursão estrita no horizonte de teste (nunca alimenta a janela com $S_{\max}$ observado do próprio horizonte).
- **Configuração centralizada:** paths, splits, lags, janela de rolling, épocas, batch e grades de HPT vivem em `config.py`. **Editar apenas este arquivo** para alterar experimentos.

**Ambiente de referência** (tempos reportados no Capítulo 5): AMD Ryzen 7 5700G, 8 núcleos / 16 threads, sem aceleração por GPU.

## Principais achados

Os quatro achados centrais da dissertação:

1. **Suficiência do histórico univariado é propriedade arquitetural, e rara.** Sob recursão estrita de um ano, apenas o **N-BEATS** sustentou previsões UV competitivas nos três transformadores — vencedor isolado no perfil mais estocástico (T3).
2. **Ganho de atributos exógenos depende do algoritmo.** Crítico para modelos baseados em árvores e para o CNN-LSTM; condicionado ao conteúdo informativo no LSTM; praticamente nulo no SVR.
3. **Escolha do melhor modelo depende do perfil do ativo.** Não há vencedor único: T1 favorece N-BEATS-MV e boosting-MV; T2 favorece boosting-MV; T3 favorece N-BEATS-UV.
4. **Critério de seleção deve coincidir com a tarefa.** A HPT por MAE recursivo de 365 passos, uniforme entre famílias, é decisiva para comparações justas — o critério clássico de 1 passo à frente favorece configurações que degeneram sob recursão longa.

## Limitações

Limitações declaradas na Seção 6.4 da dissertação, relevantes para interpretação e uso operacional:

- Escopo geográfico restrito a João Pessoa; três transformadores de perfis representativos, mas não exaustivos.
- Teste em um único ano civil (2024); ciclos plurianuais estruturais não são plenamente capturados.
- Ausência de dados operacionais desagregados (geração distribuída fotovoltaica, manobras, cronograma de manutenção).
- Atributos climáticos são normais climatológicas históricas, não previsões operacionais.
- Grade de HPT deliberadamente estreita por custo do critério recursivo; otimização bayesiana é direção natural.
- Arquiteturas baseadas em atenção (TFT, N-HiTS, PatchTST) não foram avaliadas.

## Como citar

**Dissertação:**
```bibtex
@mastersthesis{Santos2026,
  author  = {Santos, Ruan Carlos Virginio dos},
  title   = {Previs{\~a}o de Demanda El{\'e}trica em Transformadores de Distribui{\c{c}}{\~a}o Urbana:
             Um Estudo Comparativo de Previs{\~a}o de M{\'e}dio Prazo com Aprendizado de M{\'a}quina},
  school  = {Instituto Federal de Educa{\c{c}}{\~a}o, Ci{\^e}ncia e Tecnologia da Para{\'i}ba (IFPB)},
  address = {Jo{\~a}o Pessoa, PB, Brasil},
  year    = {2026},
  type    = {Disserta{\c{c}}{\~a}o de Mestrado},
  note    = {Programa de P{\'o}s-Gradua{\c{c}}{\~a}o em Tecnologia da Informa{\c{c}}{\~a}o}
}
```

**Dataset e experimentos:**
```bibtex
@misc{SantosPessoaMoura2025,
  author = {Santos, Ruan C. V. and Pessoa, Diego E. R. and Moura, Thiago J. M.},
  title  = {Electricity Demand at Power Transformers — Dataset and Experiments},
  year   = {2025},
  howpublished = {GitHub repository},
  url    = {https://github.com/ruanvcs/load-forecasting-power-transformers}
}
```

## Licença

Código distribuído sob licença **MIT** (ver `LICENSE`). Os dados são disponibilizados em versão anonimizada para fins estritamente acadêmicos, sujeitos a acordo de uso com a concessionária local.

---

**Contato:** dúvidas, correções ou colaborações — abrir *issue* neste repositório ou entrar em contato pelo e-mail institucional do PPGTI/IFPB.
