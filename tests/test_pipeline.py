"""Testes automatizados: leitura, ausência de vazamento e funcionamento do pipeline.

Execute com:  python -m pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src import data_processing as dp
from src import feature_engineering as fe
from src import train


def _linha(**kw):
    base = {
        "id": "1", "data_inversa": "2024-03-10", "dia_semana": "domingo", "horario": "23:30:00", "uf": "BA", "br": "116",
        "km": "450,5", "municipio": "X", "causa_acidente": "Velocidade Incompatível", "tipo_acidente": "Colisão frontal",
        "classificacao_acidente": "Com Vítimas Fatais", "fase_dia": "Plena Noite", "sentido_via": "Crescente",
        "condicao_metereologica": "Chuva", "tipo_pista": "Simples", "tracado_via": "Curva;Declive", "uso_solo": "Não",
        "pessoas": "3", "mortos": "1", "feridos_leves": "1", "feridos_graves": "0", "ilesos": "1", "ignorados": "0",
        "feridos": "1", "veiculos": "2", "latitude": "-12,5", "longitude": "-39.1", "regional": "R", "delegacia": "D", "uop": "U",
    }
    base.update(kw)
    return base


def test_limpeza_basica():
    df = pd.DataFrame([_linha(), _linha(id="2", classificacao_acidente="Sem Vítimas", mortos="0", feridos="0",
                                        feridos_leves="0", ilesos="3", tracado_via="Não Informado")])
    out, log = dp.limpar(df)
    assert len(out) == 2
    assert out.loc[out.id == "1", config.TARGET].item() == 2
    assert out.loc[out.id == "1", "trc_curva"].item() == 1
    assert out.loc[out.id == "2", "trc_nao_informado"].item() == 1
    # decimal com vírgula e com ponto
    assert out["latitude"].iloc[0] == pytest.approx(-12.5)
    assert out["longitude"].iloc[0] == pytest.approx(-39.1)
    assert out["km"].iloc[0] == pytest.approx(450.5)
    assert out["faixa_horaria"].iloc[0] == "noite"
    assert out["trecho"].iloc[0] == "BA-116-450"


def test_alvo_invalido_descartado():
    df = pd.DataFrame([_linha(), _linha(id="3", classificacao_acidente="Ignorado")])
    out, log = dp.limpar(df)
    assert log["sem_alvo_valido"] == 1


@pytest.mark.parametrize("cenario", list(config.CENARIOS))
def test_cenarios_sem_colunas_de_desfecho(cenario):
    feats = set(config.features_do_cenario(cenario))
    assert not feats & set(config.COLUNAS_DESFECHO), "coluna de desfecho usada como variável"


def test_causa_so_no_cenario_c():
    assert "causa_acidente" not in config.features_do_cenario("A")
    assert "causa_acidente" not in config.features_do_cenario("B")
    assert "tipo_acidente" not in config.features_do_cenario("B")
    assert "causa_acidente" in config.features_do_cenario("C")


def test_checagem_de_vazamento_dispara():
    with pytest.raises(RuntimeError):
        train._checar_vazamento(["uf", "mortos"])


def test_anos_de_treino_validacao_teste_disjuntos():
    assert config.ANO_VALIDACAO not in config.ANOS_TREINO
    assert config.ANO_TESTE not in config.ANOS_TREINO
    assert config.ANO_TESTE != config.ANO_VALIDACAO
    assert max(config.ANOS_TREINO) < config.ANO_VALIDACAO < config.ANO_TESTE


def test_pipeline_treina_e_preve():
    rng = np.random.default_rng(0)
    linhas = []
    for i in range(400):
        c = rng.choice(config.CLASSES, p=[0.2, 0.7, 0.1])
        linhas.append(_linha(id=str(i), classificacao_acidente=c, horario=f"{rng.integers(0, 24):02d}:00:00",
                             tipo_acidente=rng.choice(["Colisão frontal", "Colisão traseira"]),
                             uf=rng.choice(["BA", "SP"]), km=str(rng.integers(0, 500))))
    df, _ = dp.limpar(pd.DataFrame(linhas))
    for nome, params in [("lr", {"C": 1.0}), ("xgb", {"n_estimators": 20, "max_depth": 3}), ("rf", {"n_estimators": 20})]:
        pipe = train.ajustar(nome, "A", {**params, "pesos": "raiz"}, df)
        proba = train.prever(pipe, df, "A")
        assert proba.shape == (len(df), 3)
        assert np.allclose(proba.sum(1), 1, atol=1e-5)


def test_ajuste_limiar():
    y = np.array([0, 1, 1, 1, 2, 2])
    proba = np.array([[.4, .6, 0], [.1, .8, .1], [.2, .7, .1], [.1, .8, .1], [.1, .5, .4], [.1, .6, .3]])
    aj = fe.AjusteLimiar().fit(proba, y)
    assert aj.f1_validacao_ >= 0.99  # pesos recuperam as classes raras neste exemplo separável
