"""Obtenção, leitura, harmonização e limpeza dos microdados de acidentes da PRF.

Uso (na raiz do repositório):

    python -m src.data_processing --baixar      # baixa 2017-2025 do Google Drive oficial
    python -m src.data_processing               # lê data/raw, limpa e grava o Parquet

Unidade de análise: o acidente (arquivo "Agrupados por ocorrência", datatran<ano>.csv).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

# ---------------------------------------------------------------------------
# 1. Download (fonte oficial)
# ---------------------------------------------------------------------------

def baixar_ano(ano: int, destino: Path = config.RAW_DIR) -> Path:
    """Baixa o ZIP oficial de um ano e extrai o CSV em ``data/raw``.

    Requer acesso ao Google Drive (pacote ``gdown``). Os links vêm da página de
    Dados Abertos da PRF (ver ``config.PRF_DRIVE_IDS``).
    """
    import gdown  # import tardio: só é necessário para baixar

    destino.mkdir(parents=True, exist_ok=True)
    zip_path = destino / f"datatran{ano}.zip"
    gdown.download(id=config.PRF_DRIVE_IDS[ano], output=str(zip_path), quiet=False)
    with zipfile.ZipFile(zip_path) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        zf.extract(csvs[0], destino)
        (destino / csvs[0]).rename(destino / f"datatran{ano}.csv")
    zip_path.unlink()
    return destino / f"datatran{ano}.csv"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def gravar_manifest(anos: list[int], origem: str) -> Path:
    """Registra tamanho, hash e origem de cada arquivo bruto (reprodutibilidade)."""
    itens = []
    for ano in anos:
        p = config.RAW_DIR / f"datatran{ano}.csv"
        if p.exists():
            itens.append({"ano": ano, "arquivo": p.name, "bytes": p.stat().st_size, "sha256": _sha256(p)})
    manifest = {
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "origem": origem,
        "arquivos": itens,
    }
    out = config.RAW_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# 2. Leitura
# ---------------------------------------------------------------------------

def _num(s: pd.Series) -> pd.Series:
    """Converte texto numérico aceitando vírgula ou ponto como decimal.

    Necessário porque o separador decimal de latitude/longitude muda entre anos
    (vírgula em 2017-2020, 2022, 2023, 2025; ponto em 2021 e 2024).
    """
    return pd.to_numeric(s.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")


def ler_ano(ano: int, raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """Lê o CSV de um ano (Latin-1, separador ';') com todas as colunas como texto."""
    path = raw_dir / f"datatran{ano}.csv"
    df = pd.read_csv(path, sep=";", encoding="latin-1", dtype=str, keep_default_na=False, na_values=[""])
    df["ano_arquivo"] = ano
    return df


def ler_anos(anos: list[int] = config.ANOS, raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    return pd.concat([ler_ano(a, raw_dir) for a in anos], ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Harmonização de categorias entre anos
# ---------------------------------------------------------------------------

# Renomeações observadas entre 2017 e 2025 (ver notebooks/02).
MAPA_TIPO_ACIDENTE = {
    "Colisão com objeto em movimento": "Colisão com objeto",  # 2017-2020
    "Colisão com objeto estático": "Colisão com objeto",      # 2017-2020
    "Danos eventuais": "Eventos atípicos",                     # 2017-2020
    "Sinistro pessoal de trânsito": "Eventos atípicos",        # 2024-2025, 7 casos/ano
    # "Colisão lateral" (2017-2021, sem sentido) é mantida como categoria própria:
    # juntar com "mesmo sentido"/"sentido oposto" apagaria uma diferença real de
    # letalidade (2,7% x 9,9% de acidentes fatais em 2025).
}
MAPA_CLIMA = {
    "Ignorado": np.nan,
    "Granizo": "Outros",
    "Neve": "Outros",
}
NAO_INFORMADO = {"Não Informado", "Ignorado", "(null)", "NA", ""}

DIAS = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]


def _feriados(anos) -> set:
    import holidays

    br = holidays.Brazil(years=list(anos))  # apenas feriados nacionais
    return {pd.Timestamp(d) for d in br.keys()}


def limpar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Limpa e harmoniza a base. Retorna (base limpa, log de tratamento)."""
    log: dict[str, int] = {"linhas_lidas": len(df)}
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].str.strip()

    # --- alvo -------------------------------------------------------------
    df = df[df[config.TARGET_RAW].isin(config.CLASSES)]
    log["sem_alvo_valido"] = log["linhas_lidas"] - len(df)
    df[config.TARGET] = df[config.TARGET_RAW].map({c: i for i, c in enumerate(config.CLASSES)}).astype(int)

    # --- duplicatas -------------------------------------------------------
    antes = len(df)
    df = df.drop_duplicates(subset="id", keep="first")
    log["duplicatas_id"] = antes - len(df)

    # --- datas e horas ----------------------------------------------------
    df["data"] = pd.to_datetime(df["data_inversa"], errors="coerce", format="mixed", dayfirst=False)
    hm = df["horario"].str.extract(r"^(\d{1,2}):(\d{2})")
    df["hora"] = pd.to_numeric(hm[0], errors="coerce") + pd.to_numeric(hm[1], errors="coerce") / 60
    antes = len(df)
    df = df[df["data"].notna() & df["hora"].between(0, 24, inclusive="left")]
    log["data_ou_hora_invalida"] = antes - len(df)
    df["ano"] = df["data"].dt.year.astype(int)
    df["mes"] = df["data"].dt.month.astype(int)
    df["dia_semana"] = df["data"].dt.dayofweek.map(dict(enumerate(DIAS)))  # recalculado da data
    df["fim_de_semana"] = (df["data"].dt.dayofweek >= 5).astype(int)
    feriados = _feriados(range(df["ano"].min() - 1, df["ano"].max() + 2))
    df["feriado"] = df["data"].isin(feriados).astype(int)
    viz = {d + pd.Timedelta(days=k) for d in feriados for k in (-1, 1)}
    df["vespera_ou_pos_feriado"] = (df["data"].isin(viz) & (df["feriado"] == 0)).astype(int)
    df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
    df["faixa_horaria"] = pd.cut(df["hora"], [0, 6, 12, 18, 24], right=False,
                                 labels=["madrugada", "manha", "tarde", "noite"]).astype(str)

    # --- numéricas --------------------------------------------------------
    for c in ["km", "latitude", "longitude", "pessoas", "veiculos", "mortos", "feridos",
              "feridos_leves", "feridos_graves", "ilesos", "ignorados"]:
        df[c] = _num(df[c])
    fora = ~(df["latitude"].between(-34.5, 5.5) & df["longitude"].between(-74.5, -28.5))
    log["coordenadas_fora_do_brasil"] = int(fora.sum())
    df.loc[fora, ["latitude", "longitude"]] = np.nan
    df["veiculos"] = df["veiculos"].clip(upper=10)
    df["pessoas"] = df["pessoas"].clip(upper=20)

    # --- categóricas ------------------------------------------------------
    df["br"] = _num(df["br"]).astype("Int64").astype(str).replace("<NA>", np.nan)
    df["regiao"] = df["uf"].map(config.UF_REGIAO)
    df["tipo_acidente"] = df["tipo_acidente"].replace(MAPA_TIPO_ACIDENTE)
    df["condicao_meteo"] = df["condicao_metereologica"].replace(MAPA_CLIMA)
    df["sentido_via"] = df["sentido_via"].where(~df["sentido_via"].isin(NAO_INFORMADO))
    df["causa_acidente"] = df["causa_acidente"].where(~df["causa_acidente"].isin(NAO_INFORMADO))
    for c in ["fase_dia", "tipo_pista", "uso_solo"]:
        df[c] = df[c].where(~df[c].isin(NAO_INFORMADO))

    # --- traçado da via (multivalorado: "Reta;Declive") --------------------
    tracado = df["tracado_via"].fillna("").str.replace("Interseção de vias", "Interseção de Vias", regex=False)
    tokens = tracado.str.split(";")
    for tok, col in config.TRACADO_TOKENS.items():
        df[col] = tokens.apply(lambda ts, t=tok: int(t in ts))
    df["trc_nao_informado"] = tokens.apply(lambda ts: int(ts == [""] or "Não Informado" in ts))

    # --- trecho de 10 km (chave geográfica) ------------------------------
    km10 = (np.floor(df["km"] / 10) * 10).astype("Int64").astype(str)
    df["trecho"] = df["uf"].fillna("?") + "-" + df["br"].fillna("?") + "-" + km10.replace("<NA>", "?")

    # --- inconsistência documentada --------------------------------------
    soma = df[["ilesos", "feridos_leves", "feridos_graves", "mortos", "ignorados"]].sum(axis=1)
    log["pessoas_diferente_da_soma"] = int((soma != df["pessoas"]).sum())

    log["linhas_finais"] = len(df)
    df = df.sort_values(["data", "hora", "id"]).reset_index(drop=True)
    return df, log


COLUNAS_SAIDA = (
    ["id", "data", "ano", "hora", "municipio"]
    + sorted(set(sum((config.features_do_cenario(c) for c in config.CENARIOS), [])))
    + [c for c in config.COLUNAS_DESFECHO if c not in ("classificacao_acidente",)]
    + ["classificacao_acidente", "tracado_via"]
)


def construir_base(anos: list[int] = config.ANOS, salvar: bool = True) -> tuple[pd.DataFrame, dict]:
    """Pipeline completo: lê os anos, limpa e grava ``data/processed``."""
    bruto = ler_anos(anos)
    base, log = limpar(bruto)
    cols = [c for c in dict.fromkeys(COLUNAS_SAIDA) if c in base.columns]
    base = base[cols]
    if salvar:
        base.to_parquet(config.PROCESSED_FILE, index=False)
        (config.TABLES_DIR / "log_limpeza.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return base, log


def carregar_base() -> pd.DataFrame:
    """Lê a base processada (gera se ainda não existir)."""
    if not config.PROCESSED_FILE.exists():
        construir_base()
    return pd.read_parquet(config.PROCESSED_FILE)


def resumo_historico(raw_dir: Path, anos) -> pd.DataFrame:
    """Contagem de acidentes por classe e ano (usado no gráfico do viés de seleção)."""
    linhas = []
    for a in anos:
        p = raw_dir / f"datatran{a}.csv"
        if not p.exists():
            continue
        s = pd.read_csv(p, sep=";", encoding="latin-1", usecols=[config.TARGET_RAW], dtype=str)[config.TARGET_RAW].str.strip()
        vc = s.value_counts()
        linhas.append({"ano": a, **{c: int(vc.get(c, 0)) for c in config.CLASSES}, "outros_ou_nulos": int(len(s) - vc.reindex(config.CLASSES).fillna(0).sum())})
    return pd.DataFrame(linhas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baixar", action="store_true", help="baixa os anos de config.ANOS da fonte oficial")
    ap.add_argument("--anos", type=int, nargs="*", default=config.ANOS)
    args = ap.parse_args()
    if args.baixar:
        for a in args.anos:
            print("baixando", a)
            baixar_ano(a)
        gravar_manifest(args.anos, origem="Google Drive oficial da PRF (config.PRF_DRIVE_IDS)")
    base, log = construir_base(args.anos)
    print(json.dumps(log, indent=2))
    print(base.groupby("ano")[config.TARGET].value_counts(normalize=True).unstack().round(3))


if __name__ == "__main__":
    main()
