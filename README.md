# Scripts pra dissertação de mestrado

## Código pronto: Recursive Forecasting (T21a, T22a, T70)

- filtra os transformadores `T21a`, `T22a` e `T70`;
- faz forecast recursivo de séries temporais;
- roda os modelos `LGBM`, `XGBoost` e `GradientBoosting`;
- calcula e imprime MAE por transformador/modelo.

Arquivo: `recursive_forecasting_multimodel.py`

### Como rodar

```bash
python recursive_forecasting_multimodel.py
```

Base usada por padrão:

- `bases_tratadas/daily_peak_transformers_dataset.csv`
