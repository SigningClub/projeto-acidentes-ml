"""Treinamento, ajuste de hiperparâmetros e experimentos.

Protocolo (ver README, seção "Metodologia"):

1. ``janela``  - escolhe o ano inicial do treino olhando só a validação (2024).
2. ``busca``   - busca aleatória de hiperparâmetros: treina em 2019-2023, mede em 2024.
3. ``final``   - para cada modelo e cenário: treina em 2019-2023, ajusta o limiar de
                 decisão em 2024, retreina em 2019-2024 e prevê 2025 UMA vez.
4. ``cv``      - validação cruzada temporal (origem móvel) com os melhores parâmetros.
5. ``extras``  - split aleatório x temporal (efeito de um protocolo otimista).

Uso:

    python -m src.train --etapa tudo
    python -m src.train --etapa busca --modelos xgb
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src import config
from src import evaluate as ev
from src import feature_engineering as fe
from src.data_processing import carregar_base

PRED_TESTE = config.REPORTS_DIR / "predicoes_teste_2025.parquet"
PARAMS_FILE = config.TABLES_DIR / "melhores_params.json"

# ---------------------------------------------------------------------------
# Espaços de busca (amostrados com ParameterSampler)
# ---------------------------------------------------------------------------
ESPACOS = {
    "lr": {"C": [0.01, 0.1, 1.0, 10.0], "pesos": ["balanceado", "raiz", "nenhum"]},
    "rf": {
        "n_estimators": [200],
        "max_depth": [None, 20, 30],
        "min_samples_leaf": [5, 10, 25],
        "max_features": ["sqrt", 0.3],
        "pesos": ["balanceado", "raiz", "nenhum"],
    },
    "xgb": {
        "n_estimators": [300, 500, 800],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [4, 6, 8],
        "min_child_weight": [1, 5, 20],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [1.0, 5.0],
        "pesos": ["balanceado", "raiz", "nenhum"],
    },
}
N_ITER = {"lr": 8, "rf": 6, "xgb": 12}
PARAMS_PADRAO_XGB = {"n_estimators": 400, "learning_rate": 0.08, "max_depth": 6, "subsample": 0.8,
                     "colsample_bytree": 0.8, "pesos": "raiz"}


# ---------------------------------------------------------------------------
# Construção dos modelos
# ---------------------------------------------------------------------------

class DummyPrior(ClassifierMixin, BaseEstimator):
    """Referência mínima: probabilidade = frequência das classes no treino;
    classe prevista = sorteio com essas frequências (estratégia 'stratified')."""

    def fit(self, X, y, sample_weight=None):
        self.prior_ = np.bincount(y, minlength=3) / len(y)
        self.classes_ = np.arange(3)
        return self

    def predict_proba(self, X):
        return np.tile(self.prior_, (len(X), 1))

    def predict(self, X):
        return np.random.default_rng(config.SEED).choice(3, size=len(X), p=self.prior_)


def pesos_amostra(y: np.ndarray, modo: str) -> np.ndarray | None:
    """'balanceado' = inverso da frequência; 'raiz' = raiz do inverso (meio-termo)."""
    if modo == "nenhum":
        return None
    w = compute_sample_weight("balanced", y)
    return np.sqrt(w) if modo == "raiz" else w


def construir(nome: str, cenario: str, params: dict) -> Pipeline:
    p = {k: v for k, v in params.items() if k != "pesos"}
    if nome == "dummy":
        return Pipeline([("clf", DummyPrior())])
    if nome == "lr":
        clf = LogisticRegression(max_iter=1000, random_state=config.SEED, **p)
        return Pipeline([("pre", fe.preprocessador_linear(cenario)), ("clf", clf)])
    if nome == "rf":
        clf = RandomForestClassifier(n_jobs=-1, random_state=config.SEED, **p)
        return Pipeline([("pre", fe.preprocessador_arvores(cenario)), ("clf", clf)])
    if nome == "xgb":
        clf = XGBClassifier(objective="multi:softprob", tree_method="hist", n_jobs=-1,
                            random_state=config.SEED, eval_metric="mlogloss", **p)
        return Pipeline([("pre", fe.preprocessador_arvores(cenario)), ("clf", clf)])
    raise ValueError(nome)


def ajustar(nome: str, cenario: str, params: dict, df: pd.DataFrame) -> Pipeline:
    X, y = df[config.features_do_cenario(cenario)], df[config.TARGET].to_numpy()
    _checar_vazamento(X.columns)
    pipe = construir(nome, cenario, params)
    w = pesos_amostra(y, params.get("pesos", "nenhum"))
    kw = {} if w is None else {"clf__sample_weight": w}
    return pipe.fit(X, y, **kw)


def prever(pipe: Pipeline, df: pd.DataFrame, cenario: str) -> np.ndarray:
    return pipe.predict_proba(df[config.features_do_cenario(cenario)])


def _checar_vazamento(colunas) -> None:
    proibidas = set(colunas) & set(config.COLUNAS_DESFECHO)
    if proibidas:
        raise RuntimeError(f"Vazamento: colunas de desfecho nas features: {sorted(proibidas)}")


# ---------------------------------------------------------------------------
# Dados e divisões
# ---------------------------------------------------------------------------

def dividir(base: pd.DataFrame, anos_treino=None, ano_val=config.ANO_VALIDACAO, ano_teste=config.ANO_TESTE):
    anos_treino = anos_treino or config.ANOS_TREINO
    return (base[base.ano.isin(anos_treino)], base[base.ano == ano_val], base[base.ano == ano_teste])


def _resumo(y, proba, ajuste: fe.AjusteLimiar | None = None) -> dict:
    pred = proba.argmax(1)
    m = {"f1_macro_argmax": ev.f1_macro_rapido(y, pred),
         "pr_auc_fatais": float(ev.average_precision_score((y == 2).astype(int), proba[:, 2])),
         "roc_auc_macro": float(ev.roc_auc_score(y, proba, multi_class="ovr"))}
    if ajuste is not None:
        m["f1_macro_ajustado"] = ev.f1_macro_rapido(y, ajuste.predict(proba))
    return m


# ---------------------------------------------------------------------------
# Etapa 1: janela de treino
# ---------------------------------------------------------------------------

def experimento_janela(base: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for inicio in (2017, 2019, 2021):
        tr, va, _ = dividir(base, list(range(inicio, 2024)))
        t = time.time()
        pipe = ajustar("xgb", "A", PARAMS_PADRAO_XGB, tr)
        proba = prever(pipe, va, "A")
        aj = fe.AjusteLimiar().fit(proba, va[config.TARGET].to_numpy())
        y = va[config.TARGET].to_numpy()
        linhas.append({"inicio_treino": inicio, "n_treino": len(tr), **_resumo(y, proba, aj),
                       "pct_prev_sem_vitimas_argmax": 100 * (proba.argmax(1) == 0).mean(),
                       "pct_real_sem_vitimas": 100 * (y == 0).mean(), "segundos": time.time() - t})
        print(linhas[-1], flush=True)
    df = pd.DataFrame(linhas)
    df.to_csv(config.TABLES_DIR / "janela_treino.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Etapa 2: busca de hiperparâmetros (cenário A)
# ---------------------------------------------------------------------------

def busca(base: pd.DataFrame, nome: str, cenario: str = "A") -> dict:
    tr, va, _ = dividir(base)
    y = va[config.TARGET].to_numpy()
    linhas = []
    for i, params in enumerate(ParameterSampler(ESPACOS[nome], n_iter=N_ITER[nome], random_state=config.SEED)):
        t = time.time()
        proba = prever(ajustar(nome, cenario, params, tr), va, cenario)
        aj = fe.AjusteLimiar().fit(proba, y)
        linhas.append({"config": i, **{k: str(v) for k, v in params.items()}, **_resumo(y, proba, aj),
                       "w_sem": aj.w_[0], "w_fatais": aj.w_[2], "segundos": round(time.time() - t, 1)})
        print(nome, linhas[-1], flush=True)
    df = pd.DataFrame(linhas).sort_values("f1_macro_ajustado", ascending=False)
    df.to_csv(config.TABLES_DIR / f"busca_{nome}.csv", index=False)
    melhor = ParameterSampler(ESPACOS[nome], n_iter=N_ITER[nome], random_state=config.SEED)
    melhor = list(melhor)[int(df.iloc[0]["config"])]
    todos = json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE.exists() else {}
    todos[nome] = melhor
    PARAMS_FILE.write_text(json.dumps(todos, indent=2, default=str), encoding="utf-8")
    return melhor


def carregar_params() -> dict:
    p = json.loads(PARAMS_FILE.read_text())
    p["dummy"] = {}
    return p


# ---------------------------------------------------------------------------
# Etapa 3: modelos finais (teste 2025 usado uma única vez por modelo/cenário)
# ---------------------------------------------------------------------------

EXPERIMENTOS_FINAIS = [(m, c) for c in ("A", "B") for m in ("dummy", "lr", "rf", "xgb")] + [("xgb", "C"), ("xgb", "A_pessoas")]


def treinar_final(base: pd.DataFrame, nome: str, cenario: str, params: dict, salvar_modelo: bool = True) -> dict:
    tr, va, te = dividir(base)
    yv, yt = va[config.TARGET].to_numpy(), te[config.TARGET].to_numpy()
    # (a) treino 2019-2023 -> probabilidades na validação -> ajuste do limiar
    proba_val = prever(ajustar(nome, cenario, params, tr), va, cenario)
    aj = fe.AjusteLimiar().fit(proba_val, yv) if nome != "dummy" else fe.AjusteLimiar(grade=[1.0]).fit(proba_val, yv)
    # (b) retreino com 2019-2024 -> previsão do teste 2025
    pipe = ajustar(nome, cenario, params, pd.concat([tr, va]))
    proba = prever(pipe, te, cenario)
    pred_argmax = pipe.predict(te[config.features_do_cenario(cenario)]) if nome == "dummy" else proba.argmax(1)
    pred_aj = pred_argmax if nome == "dummy" else aj.predict(proba)
    if salvar_modelo and nome in ("lr", "xgb"):
        joblib.dump({"pipeline": pipe, "limiar": aj.w_, "cenario": cenario, "params": params,
                     "features": config.features_do_cenario(cenario), "anos_treino": sorted(set(tr.ano) | set(va.ano))},
                    config.MODELS_DIR / f"{nome}_{cenario}.joblib", compress=3)
    return {"proba": proba, "pred_argmax": pred_argmax, "pred": pred_aj, "w": aj.w_.tolist(),
            "validacao": _resumo(yv, proba_val, aj), "y": yt}


def etapa_final(base: pd.DataFrame, experimentos=None) -> None:
    params = carregar_params()
    _, _, te = dividir(base)
    cols = ["id", "data", "ano", "mes", "hora", "uf", "regiao", "br", "km", "trecho", "fase_dia", "tipo_acidente",
            "condicao_meteo", "tipo_pista", "uso_solo", "veiculos", config.TARGET]
    pred = pd.read_parquet(PRED_TESTE) if PRED_TESTE.exists() else te[cols].rename(columns={config.TARGET: "y"}).reset_index(drop=True)
    resumo_path = config.TABLES_DIR / "resumo_validacao.json"
    resumo = json.loads(resumo_path.read_text()) if resumo_path.exists() else {}
    for nome, cen in experimentos or EXPERIMENTOS_FINAIS:
        t = time.time()
        r = treinar_final(base, nome, cen, params[nome])
        k = f"{nome}_{cen}"
        for j in range(3):
            pred[f"{k}__p{j}"] = r["proba"][:, j]
        pred[f"{k}__pred"] = r["pred"]
        pred[f"{k}__pred_argmax"] = r["pred_argmax"]
        resumo[k] = {"validacao_2024": r["validacao"], "pesos_limiar": r["w"], "params": params[nome]}
        print(k, r["validacao"], f"{time.time() - t:.0f}s", flush=True)
        pred.to_parquet(PRED_TESTE, index=False)
        resumo_path.write_text(json.dumps(resumo, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Etapa 4: validação cruzada temporal
# ---------------------------------------------------------------------------

def validacao_cruzada_temporal(base: pd.DataFrame, modelos=("lr", "rf", "xgb"), cenario="A") -> pd.DataFrame:
    params = carregar_params()
    linhas = []
    for nome in modelos:
        for anos_tr, ano_va in config.FOLDS_TEMPORAIS:
            tr, va = base[base.ano.isin(anos_tr)], base[base.ano == ano_va]
            proba = prever(ajustar(nome, cenario, params[nome], tr), va, cenario)
            linhas.append({"modelo": nome, "treino": f"{anos_tr[0]}-{anos_tr[-1]}", "validacao": ano_va,
                           **_resumo(va[config.TARGET].to_numpy(), proba)})
            print(linhas[-1], flush=True)
    df = pd.DataFrame(linhas)
    df.to_csv(config.TABLES_DIR / "cv_temporal.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Etapa 5: split aleatório x temporal
# ---------------------------------------------------------------------------

def experimento_split_aleatorio(base: pd.DataFrame) -> dict:
    params = carregar_params()["xgb"]
    dados = base[base.ano >= config.ANO_INICIO_TREINO]
    tr, te = train_test_split(dados, test_size=0.2, stratify=dados[config.TARGET], random_state=config.SEED)
    proba = prever(ajustar("xgb", "A", params, tr), te, "A")
    r = {"split": "aleatório estratificado 80/20 (2019-2025)", "n_teste": len(te),
         **_resumo(te[config.TARGET].to_numpy(), proba)}
    ev.salvar_json(r, "split_aleatorio.json")
    print(r)
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--etapa", default="tudo", choices=["janela", "busca", "final", "cv", "extras", "tudo"])
    ap.add_argument("--modelos", nargs="*", default=["lr", "rf", "xgb"])
    args = ap.parse_args()
    base = carregar_base()
    if args.etapa in ("janela", "tudo"):
        experimento_janela(base)
    if args.etapa in ("busca", "tudo"):
        for m in args.modelos:
            busca(base, m)
    if args.etapa in ("final", "tudo"):
        etapa_final(base)
    if args.etapa in ("cv", "tudo"):
        validacao_cruzada_temporal(base, args.modelos)
    if args.etapa in ("extras", "tudo"):
        experimento_split_aleatorio(base)


if __name__ == "__main__":
    main()
