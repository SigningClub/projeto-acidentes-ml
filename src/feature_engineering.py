"""Transformações de atributos, montadas como ``ColumnTransformer`` do scikit-learn.

As variáveis derivadas de cada linha (hora cíclica, feriado, traçado, trecho) são
criadas em ``data_processing.limpar`` porque dependem só da própria linha e não
aprendem nada dos dados. Tudo o que APRENDE com os dados (categorias frequentes,
médias, escalas, target encoding) fica aqui, dentro do ``Pipeline``, para ser
ajustado apenas no conjunto de treino de cada fold. Isso evita vazamento.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

from src import config

MIN_FREQ = 50  # categorias com menos de 50 casos no treino viram "infrequente"


def _target_encoder() -> TargetEncoder:
    """Target encoding multiclasse do trecho (UF-BR-faixa de 10 km).

    Em ``fit_transform`` o scikit-learn usa *cross fitting* (5 folds): a codificação
    de cada linha de treino é calculada sem a própria linha, o que evita que o
    modelo "decore" o alvo. Na validação e no teste usa as médias de todo o treino.
    """
    return TargetEncoder(target_type="multiclass", smooth="auto", cv=5, shuffle=True, random_state=config.SEED)


def preprocessador_linear(cenario: str) -> ColumnTransformer:
    """One-Hot + padronização: necessário para a Regressão Logística."""
    c = config.CENARIOS[cenario]
    cat = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="ausente")),
        ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=MIN_FREQ)),
    ])
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())])
    te = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="ausente")),
        ("te", _target_encoder()),
        ("esc", StandardScaler()),
    ])
    return ColumnTransformer([
        ("cat", cat, c["cat"]),
        ("num", num, c["num"]),
        ("te", te, c["te"]),
    ], sparse_threshold=0.3, verbose_feature_names_out=True)


def preprocessador_arvores(cenario: str) -> ColumnTransformer:
    """Codificação ordinal, sem escala: árvores dividem por limiares e não precisam.

    Valores ausentes ficam como NaN (Random Forest e XGBoost tratam nativamente).
    """
    c = config.CENARIOS[cenario]
    cat = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1,
        encoded_missing_value=-2, min_frequency=MIN_FREQ, dtype=np.float32,
    )
    te = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="ausente")),
        ("te", _target_encoder()),
    ])
    return ColumnTransformer([
        ("cat", cat, c["cat"]),
        ("num", "passthrough", c["num"]),
        ("te", te, c["te"]),
    ], verbose_feature_names_out=True)


def nomes_legiveis(nomes) -> list[str]:
    """Remove prefixos do ColumnTransformer (``cat__uf`` -> ``uf``)."""
    out = []
    for n in nomes:
        n = n.split("__", 1)[-1]
        n = n.replace("trecho_0", "trecho→P(sem vítimas)").replace("trecho_1", "trecho→P(feridas)").replace("trecho_2", "trecho→P(fatais)")
        out.append(n)
    return out


class AjusteLimiar:
    """Ajuste de decisão por classe: prevê ``argmax(p * w)``.

    Os pesos ``w`` são escolhidos no conjunto de VALIDAÇÃO para maximizar o
    F1-macro, e depois aplicados sem alteração ao teste. É uma forma simples de
    deslocar a fronteira de decisão a favor das classes raras sem reamostrar dados.
    """

    def __init__(self, grade=None):
        # grade log-espaçada de 0,2 a 5 (41 valores) para as classes Sem vítimas e Fatais
        self.grade = grade if grade is not None else np.round(np.geomspace(0.2, 5.0, 41), 3)
        self.w_ = np.ones(3)

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "AjusteLimiar":
        from src.evaluate import f1_macro_rapido

        melhor, w_melhor = -1.0, np.ones(3)
        # w da classe "Feridas" fixo em 1; busca em grade para as outras duas
        for w0 in self.grade:
            for w2 in self.grade:
                w = np.array([w0, 1.0, w2])
                f = f1_macro_rapido(y, np.argmax(proba * w, axis=1))
                if f > melhor:
                    melhor, w_melhor = f, w
        self.w_ = w_melhor
        self.f1_validacao_ = melhor
        return self

    def predict(self, proba: np.ndarray) -> np.ndarray:
        return np.argmax(proba * self.w_, axis=1)
