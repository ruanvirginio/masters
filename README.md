# Scripts pra dissertação de mestrado

## Código pronto: Recursive Forecasting (T21a, T22a, T70)

Criei um script executável com o fluxo que você pediu:

- filtra os transformadores `T21a`, `T22a` e `T70`;
- faz forecast recursivo de séries temporais;
- roda os modelos `LGBM`, `XGBoost` e `GradientBoosting`;
- calcula e imprime MAE por transformador/modelo.
- também deixei a célula final pronta dentro de `Recursive_Forecasting.ipynb` com o mesmo fluxo.

Arquivo: `recursive_forecasting_multimodel.py`

### Como rodar

```bash
python recursive_forecasting_multimodel.py
```

Base usada por padrão:

- `bases_tratadas/daily_peak_transformers_dataset.csv`

## Como aceitar o PR (GitHub)

1. Abra a página do Pull Request.
2. Clique em **"Files changed"** para revisar o código.
3. Clique em **"Review changes"** (opcional) e aprove.
4. Clique em **"Merge pull request"**.
5. Confirme em **"Confirm merge"**.

Se quiser, também posso te entregar esse mesmo código dentro do notebook `Recursive_Forecasting.ipynb` célula por célula.
