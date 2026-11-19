"""Configuração central do projeto.

Tudo o que define o experimento fica aqui: caminhos, anos de corte, variáveis
permitidas em cada cenário e colunas proibidas (vazamento). Os demais módulos
importam estas constantes; nenhum deles define listas de variáveis por conta própria.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

PROCESSED_FILE = PROCESSED_DIR / "acidentes_2017_2025.parquet"

SEED = 42

# ---------------------------------------------------------------------------
# Fonte oficial: IDs dos arquivos "Agrupados por ocorrência" no Google Drive,
# copiados da página de Dados Abertos da PRF em 19/11/2026:
# https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf
# ---------------------------------------------------------------------------
PRF_DRIVE_IDS = {
    2026: "1A3IirNm0AzRaSosA1IS94DOVmvKsn0Ol",
    2025: "1-G3MdmHBt6CprDwcW99xxC4BZ2DU5ryR",
    2024: "14lB0vqMFkaZj8HZ44b0njYgxs9nAN8KO",
    2023: "1-WO3SfNrwwZ5_l7fRTiwBKRw7mi1-HUq",
    2022: "1PRQjuV5gOn_nn6UNvaJyVURDIfbSAK4-",
    2021: "12xH8LX9aN2gObR766YN3cMcuycwyCJDz",
    2020: "1esu6IiH5TVTxFoedv6DBGDd01Gvi8785",
    2019: "1pN3fn2wY34GH6cY-gKfbxRJJBFE0lb_l",
    2018: "1cM4IgGMIiR-u4gBIH5IEe3DcvBvUzedi",
    2017: "1HPLWt5f_l4RIX3tKjI4tUXyZOev52W0N",
    2016: "16qooQl_ySoW61CrtsBbreBVNPYlEkoYm",
    2015: "1DyqR5FFcwGsamSag-fGm13feQt0Y-3Da",
    2014: "1FpF5wTBsRDkEhLm3z2g8XDiXr9SO9Uk8",
    2013: "1p_7lw9RzkINfscYAZSmc-Z9Ci4ZPJyEr",
    2012: "18Yz2prqKSLthrMmW-73vrOiDmKTCL6xE",
    2011: "1HHhgLF-kSR6Gde2qOaTXL3T5ieD33hpG",
    2010: "1_yU6FRh8M7USjiChQwyF20NtY48GTmEX",
    2009: "1qkVatg0pC_zosuBs0NCSgEXDJvBbnTYC",
    2008: "1_OSeHlyKJw8cIhMS_JzSg1RlYX8k6vSG",
    2007: "1EFpZF5F6cB0DOHd2Uxnj7X948WE69a8e",
}

# Anos carregados no projeto (mesmo sistema de origem, BAT, desde 2017).
ANOS = list(range(2017, 2026))

# ---------------------------------------------------------------------------
# Separação temporal
# ---------------------------------------------------------------------------
# A janela de treino foi escolhida na validação de 2024 (ver
# src/train.py::experimento_janela): 2017-2018 ainda estão em transição após a
# declaração online de acidentes sem vítimas (e-DAT, 2015) e distorcem a classe
# "Sem Vítimas".
ANO_INICIO_TREINO = 2019
ANOS_TREINO = list(range(ANO_INICIO_TREINO, 2024))  # 2019-2023
ANO_VALIDACAO = 2024
ANO_TESTE = 2025

# Folds da validação cruzada temporal (origem móvel) dentro do treino.
FOLDS_TEMPORAIS = [
    (list(range(2019, 2021)), 2021),
    (list(range(2019, 2022)), 2022),
    (list(range(2019, 2023)), 2023),
    (list(range(2019, 2024)), 2024),
]

# ---------------------------------------------------------------------------
# Alvo
# ---------------------------------------------------------------------------
TARGET_RAW = "classificacao_acidente"
TARGET = "gravidade"
CLASSES = ["Sem Vítimas", "Com Vítimas Feridas", "Com Vítimas Fatais"]
CLASSES_CURTAS = ["Sem vítimas", "Feridas", "Fatais"]
CLASSE_FATAL = 2  # índice da classe de interesse

# ---------------------------------------------------------------------------
# Colunas proibidas: definem o alvo ou só existem depois do acidente.
# ---------------------------------------------------------------------------
COLUNAS_DESFECHO = [
    "mortos", "feridos", "feridos_leves", "feridos_graves", "ilesos", "ignorados",
    TARGET_RAW, TARGET,
]
COLUNAS_IDENTIFICACAO = ["id", "delegacia", "uop", "regional"]

# ---------------------------------------------------------------------------
# Variáveis por cenário de informação
# ---------------------------------------------------------------------------
TRACADO_TOKENS = {
    # token original -> nome da coluna binária
    "Reta": "trc_reta",
    "Curva": "trc_curva",
    "Interseção de Vias": "trc_intersecao",
    "Ponte": "trc_ponte",
    "Viaduto": "trc_viaduto",
    "Rotatória": "trc_rotatoria",
    "Retorno Regulamentado": "trc_retorno",
    "Túnel": "trc_tunel",
    "Desvio Temporário": "trc_desvio",
}
# "Aclive", "Declive" e "Em Obras" só aparecem a partir de 2022: ficam de fora
# para não criar uma variável que é sempre 0 nos anos antigos.
TRACADO_COLS = list(TRACADO_TOKENS.values()) + ["trc_nao_informado"]

CAT_CONTEXTO = [
    "uf", "regiao", "br", "dia_semana", "fase_dia", "condicao_meteo",
    "tipo_pista", "uso_solo", "sentido_via", "faixa_horaria",
]
NUM_CONTEXTO = [
    "hora_sin", "hora_cos", "mes", "km", "latitude", "longitude",
    "fim_de_semana", "feriado", "vespera_ou_pos_feriado",
] + TRACADO_COLS
# Chave de alta cardinalidade codificada por target encoding com validação cruzada.
TE_CONTEXTO = ["trecho"]

CENARIOS = {
    # B: só o contexto (tempo, local, via, clima)
    "B": {"cat": CAT_CONTEXTO, "num": NUM_CONTEXTO, "te": TE_CONTEXTO},
    # A: contexto + o que a chamada informa (tipo de acidente e nº de veículos)
    "A": {"cat": CAT_CONTEXTO + ["tipo_acidente"], "num": NUM_CONTEXTO + ["veiculos"], "te": TE_CONTEXTO},
    # C: A + causa presumível (só conhecida após a perícia) -> NÃO é previsão
    "C": {"cat": CAT_CONTEXTO + ["tipo_acidente", "causa_acidente"], "num": NUM_CONTEXTO + ["veiculos"], "te": TE_CONTEXTO},
    # A + pessoas: teste de sensibilidade
    "A_pessoas": {"cat": CAT_CONTEXTO + ["tipo_acidente"], "num": NUM_CONTEXTO + ["veiculos", "pessoas"], "te": TE_CONTEXTO},
}


def features_do_cenario(cenario: str) -> list[str]:
    """Lista ordenada de colunas de entrada de um cenário."""
    c = CENARIOS[cenario]
    return c["cat"] + c["num"] + c["te"]


UF_REGIAO = {
    "AC": "Norte", "AP": "Norte", "AM": "Norte", "PA": "Norte", "RO": "Norte", "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste", "PB": "Nordeste",
    "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste", "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste", "MS": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "RS": "Sul", "SC": "Sul",
}

for _d in (RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, MODELS_DIR, FIGURES_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
