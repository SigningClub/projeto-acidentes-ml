# Resultados (gerado por `python -m src.evaluate`)

Todos os números abaixo são calculados pelos scripts a partir de `reports/predicoes_teste_2025.parquet` e `reports/tables/`.

## Escolha da janela de treino (validação 2024, XGBoost com parâmetros padrão)

| inicio_treino | n_treino | f1_macro_ajustado | f1_macro_argmax | pr_auc_fatais | roc_auc_macro |
| --- | --- | --- | --- | --- | --- |
| 2017 | 486.737 | 0.553 | 0.551 | 0.349 | 0.777 |
| 2019 | 327.879 | 0.559 | 0.555 | 0.346 | 0.782 |
| 2021 | 196.885 | 0.558 | 0.555 | 0.339 | 0.781 |


## Validação cruzada temporal (4 folds anuais, 2021-2024, cenário A)

| modelo | f1_macro_argmax_mean | f1_macro_argmax_std | pr_auc_fatais_mean | pr_auc_fatais_std | roc_auc_macro_mean | roc_auc_macro_std |
| --- | --- | --- | --- | --- | --- | --- |
| lr | 0.438 | 0.021 | 0.304 | 0.015 | 0.750 | 0.014 |
| rf | 0.494 | 0.009 | 0.307 | 0.016 | 0.749 | 0.016 |
| xgb | 0.478 | 0.006 | 0.324 | 0.016 | 0.758 | 0.019 |


## Teste 2025 (usado uma única vez)

| modelo | cenario | f1_macro | f1_macro_ic_inf | f1_macro_ic_sup | f1_macro_argmax | recall_fatais | precision_fatais | f1_fatais | pr_auc_fatais | roc_auc_macro_ovr | acuracia | recall_fatais_com_precisao_0.30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dummy | A | 0.334 | — | — | 0.334 | 0.072 | 0.073 | 0.072 | 0.072 | 0.500 | 0.618 | 0.000 |
| lr | A | 0.543 | 0.537 | 0.547 | 0.459 | 0.382 | 0.375 | 0.378 | 0.316 | 0.768 | 0.747 | 0.505 |
| rf | A | 0.551 | 0.546 | 0.556 | 0.508 | 0.407 | 0.381 | 0.394 | 0.328 | 0.771 | 0.744 | 0.527 |
| xgb | A | 0.561 | 0.555 | 0.566 | 0.485 | 0.421 | 0.391 | 0.405 | 0.348 | 0.780 | 0.759 | 0.549 |
| xgb | A_pessoas | 0.575 | 0.570 | 0.580 | 0.498 | 0.429 | 0.393 | 0.410 | 0.349 | 0.793 | 0.765 | 0.558 |
| dummy | B | 0.334 | — | — | 0.334 | 0.072 | 0.073 | 0.072 | 0.072 | 0.500 | 0.618 | 0.000 |
| lr | B | 0.436 | 0.431 | 0.440 | 0.306 | 0.256 | 0.223 | 0.238 | 0.179 | 0.664 | 0.674 | 0.127 |
| rf | B | 0.435 | 0.431 | 0.440 | 0.412 | 0.265 | 0.215 | 0.237 | 0.178 | 0.661 | 0.672 | 0.102 |
| xgb | B | 0.442 | 0.438 | 0.447 | 0.394 | 0.290 | 0.219 | 0.250 | 0.189 | 0.667 | 0.654 | 0.132 |
| xgb | C | 0.576 | 0.571 | 0.581 | 0.506 | 0.433 | 0.402 | 0.417 | 0.368 | 0.794 | 0.762 | 0.579 |


Modelo escolhido na validação (cenário A): **xgb_A**. McNemar contra rf_A: p = 1.25e-55; contra lr_A: p = 1.1e-25.


Split aleatório (otimista) com XGBoost cenário A: F1-macro (argmax) = 0.487, PR-AUC Fatais = 0.336.


## Modelo x regra simples (classe Fatais)

| comparacao | precisao | recall | f1 | fatais_reais | previstos_fatais | acertos |
| --- | --- | --- | --- | --- | --- | --- |
| regra_frontal_ou_pedestre | 0.295 | 0.441 | 0.353 | 5209 | 7795 | 2297 |
| modelo | 0.391 | 0.421 | 0.405 | 5209 | 5620 | 2195 |
| modelo_dentro_frontal_pedestre | 0.407 | 0.850 | 0.550 | 2297 | 4801 | 1953 |
| modelo_outros_tipos | 0.295 | 0.083 | 0.130 | 2912 | 819 | 242 |


## Desempenho por região (melhor modelo, cenário A)

| regiao | n | pct_fatais | f1_macro | recall_fatais | precisao_fatais | pr_auc_fatais |
| --- | --- | --- | --- | --- | --- | --- |
| Sudeste | 23.323 | 5.6 | 0.571 | 0.360 | 0.380 | 0.308 |
| Sul | 20.715 | 5.6 | 0.553 | 0.389 | 0.378 | 0.311 |
| Nordeste | 16.018 | 10.5 | 0.550 | 0.480 | 0.395 | 0.393 |
| Centro-Oeste | 8.497 | 7.4 | 0.557 | 0.386 | 0.446 | 0.347 |
| Norte | 3.975 | 11.1 | 0.530 | 0.516 | 0.370 | 0.388 |


## Importância por permutação (top 10)

| variavel | queda_pr_auc_fatais | desvio |
| --- | --- | --- |
| tipo_acidente | 0.200 | 0.005 |
| veiculos | 0.050 | 0.003 |
| trecho | 0.035 | 0.004 |
| latitude | 0.019 | 0.003 |
| hora_cos | 0.018 | 0.002 |
| uso_solo | 0.014 | 0.001 |
| tipo_pista | 0.012 | 0.003 |
| fase_dia | 0.009 | 0.002 |
| hora_sin | 0.007 | 0.002 |
| km | 0.005 | 0.002 |


## SHAP — classe Fatais (top 10)

| variavel | media_abs_shap_fatais |
| --- | --- |
| tipo_acidente | 0.429 |
| veiculos | 0.267 |
| trecho→P(fatais) | 0.209 |
| uso_solo | 0.124 |
| hora_cos | 0.088 |
| latitude | 0.086 |
| fase_dia | 0.062 |
| condicao_meteo | 0.037 |
| longitude | 0.037 |
| uf | 0.036 |


## Regressão Logística — maiores coeficientes da classe Fatais

| variavel | coef_fatais | razao_chances |
| --- | --- | --- |
| tipo_acidente_Atropelamento de Pedestre | 2.245 | 9.436 |
| tipo_acidente_Incêndio | -2.003 | 0.135 |
| tipo_acidente_Colisão frontal | 1.296 | 3.654 |
| tipo_acidente_Engavetamento | -1.049 | 0.350 |
| tipo_acidente_Derramamento de carga | -0.779 | 0.459 |
| br_465 | -0.645 | 0.524 |
| br_405 | 0.569 | 1.767 |
| tipo_acidente_Queda de ocupante de veículo | 0.558 | 1.747 |
| sentido_via_ausente | -0.540 | 0.583 |
| br_354 | -0.527 | 0.591 |
| br_412 | 0.472 | 1.603 |
| uso_solo_Sim | -0.461 | 0.631 |
