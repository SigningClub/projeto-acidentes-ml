"""Métricas, testes estatísticos, figuras e análises de erro.

Uso (depois de ``python -m src.train``):

    python -m src.evaluate      # gera tabelas e figuras em reports/
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_recall_curve, precision_recall_fscore_support, roc_auc_score,
)

from src import config

# Paleta discreta e consistente entre figuras
CORES = {"dummy": "#9aa0a6", "lr": "#4c78a8", "rf": "#59a14f", "xgb": "#e15759"}
NOMES = {"dummy": "Dummy (estratificado)", "lr": "Regressão Logística", "rf": "Random Forest", "xgb": "XGBoost"}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "font.size": 10})


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def f1_macro_rapido(y: np.ndarray, pred: np.ndarray, k: int = 3) -> float:
    """F1-macro via matriz de confusão em NumPy (rápido para bootstrap)."""
    cm = np.bincount(k * y + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    p = tp / np.maximum(cm.sum(0), 1)
    r = tp / np.maximum(cm.sum(1), 1)
    f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)
    return float(f.mean())


def recall_com_precisao_minima(y_bin, score, precisao_min=0.30) -> float:
    p, r, _ = precision_recall_curve(y_bin, score)
    ok = p >= precisao_min
    return float(r[ok].max()) if ok.any() else 0.0


def precisao_com_recall_minimo(y_bin, score, recall_min=0.50) -> float:
    p, r, _ = precision_recall_curve(y_bin, score)
    ok = r >= recall_min
    return float(p[ok].max()) if ok.any() else 0.0


def metricas(y: np.ndarray, pred: np.ndarray, proba: np.ndarray) -> dict:
    """Todas as métricas do projeto para um conjunto de previsões."""
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    out = {
        "f1_macro": f1_score(y, pred, average="macro"),
        "f1_ponderado": f1_score(y, pred, average="weighted"),
        "acuracia": accuracy_score(y, pred),
        "roc_auc_macro_ovr": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
        "brier_medio": float(np.mean([brier_score_loss((y == k).astype(int), proba[:, k]) for k in range(3)])),
    }
    for k, nome in enumerate(["sem", "feridas", "fatais"]):
        out[f"precision_{nome}"] = pr[k]
        out[f"recall_{nome}"] = rc[k]
        out[f"f1_{nome}"] = f1[k]
        out[f"pr_auc_{nome}"] = average_precision_score((y == k).astype(int), proba[:, k])
    yf = (y == config.CLASSE_FATAL).astype(int)
    out["prevalencia_fatais"] = yf.mean()
    out["recall_fatais_com_precisao_0.30"] = recall_com_precisao_minima(yf, proba[:, 2], 0.30)
    out["precisao_fatais_com_recall_0.50"] = precisao_com_recall_minimo(yf, proba[:, 2], 0.50)
    return {k: float(v) for k, v in out.items()}


def bootstrap_ic(y, pred, proba, n=1000, seed=config.SEED) -> dict:
    """IC de 95% por bootstrap (percentis) para F1-macro, recall e PR-AUC de Fatais."""
    rng = np.random.default_rng(seed)
    m = len(y)
    f1s, recs, aps = [], [], []
    yf = (y == 2).astype(int)
    for _ in range(n):
        i = rng.integers(0, m, m)
        f1s.append(f1_macro_rapido(y[i], pred[i]))
        tp = ((pred[i] == 2) & (y[i] == 2)).sum()
        recs.append(tp / max((y[i] == 2).sum(), 1))
        aps.append(average_precision_score(yf[i], proba[i, 2]))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"f1_macro_ic95": q(f1s), "recall_fatais_ic95": q(recs), "pr_auc_fatais_ic95": q(aps)}


def mcnemar(y, pred_a, pred_b) -> dict:
    """Teste de McNemar (acerto x erro) entre dois classificadores no mesmo teste."""
    from statsmodels.stats.contingency_tables import mcnemar as _mc

    a = pred_a == y
    b = pred_b == y
    tab = [[int((a & b).sum()), int((a & ~b).sum())], [int((~a & b).sum()), int((~a & ~b).sum())]]
    r = _mc(tab, exact=False, correction=True)
    return {"tabela": tab, "estatistica": float(r.statistic), "p_valor": float(r.pvalue)}


def metricas_por_grupo(df: pd.DataFrame, col: str, y_col="y", pred_col="pred", proba_fatal_col="p_fatal", min_n=500) -> pd.DataFrame:
    linhas = []
    for g, d in df.groupby(col, observed=True):
        if len(d) < min_n:
            continue
        y, p = d[y_col].to_numpy(), d[pred_col].to_numpy()
        yf = (y == 2).astype(int)
        linhas.append({
            col: g, "n": len(d), "pct_fatais": 100 * yf.mean(),
            "f1_macro": f1_score(y, p, average="macro"),
            "recall_fatais": ((p == 2) & (y == 2)).sum() / max(yf.sum(), 1),
            "precisao_fatais": ((p == 2) & (y == 2)).sum() / max((p == 2).sum(), 1),
            "pr_auc_fatais": average_precision_score(yf, d[proba_fatal_col]) if 0 < yf.sum() < len(yf) else np.nan,
        })
    return pd.DataFrame(linhas).sort_values("n", ascending=False)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def salvar(fig, nome: str):
    path = config.FIGURES_DIR / nome
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_matriz_confusao(y, pred, titulo: str, nome: str):
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{100*cmn[i, j]:.1f}%\n({cm[i, j]:,})".replace(",", "."), ha="center", va="center",
                    color="white" if cmn[i, j] > 0.55 else "black", fontsize=9)
    ax.set_xticks(range(3), config.CLASSES_CURTAS)
    ax.set_yticks(range(3), config.CLASSES_CURTAS)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title(titulo, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="% da classe real")
    return salvar(fig, nome)


def fig_curvas_pr(y, probas: dict, nome: str, titulo: str):
    yf = (y == 2).astype(int)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for m, p in probas.items():
        if m == "dummy":  # probabilidade constante: equivale à linha do acaso
            continue
        pr, rc, _ = precision_recall_curve(yf, p[:, 2])
        ax.plot(rc, pr, color=CORES.get(m, "k"), lw=1.8, label=f"{NOMES.get(m, m)} (AP={average_precision_score(yf, p[:, 2]):.3f})")
    ax.axhline(yf.mean(), color="#9aa0a6", ls="--", lw=1, label=f"Acaso (prevalência={yf.mean():.3f})")
    ax.set_xlabel("Recall (acidentes fatais encontrados)")
    ax.set_ylabel("Precisão")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(titulo, fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    return salvar(fig, nome)


def fig_calibracao(y, probas: dict, nome: str):
    from sklearn.calibration import calibration_curve

    yf = (y == 2).astype(int)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, 1], [0, 1], color="#9aa0a6", ls="--", lw=1, label="Calibração perfeita")
    for m, p in probas.items():
        if m == "dummy":
            continue
        fr, mp = calibration_curve(yf, p[:, 2], n_bins=10, strategy="quantile")
        ax.plot(mp, fr, marker="o", ms=3, color=CORES.get(m, "k"), label=NOMES.get(m, m))
    ax.set_xlabel("Probabilidade prevista de acidente fatal")
    ax.set_ylabel("Fração observada de fatais")
    ax.set_title("Calibração — classe Fatais (teste 2025)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    return salvar(fig, nome)


def fig_barras_horizontais(serie: pd.Series, titulo: str, xlabel: str, nome: str, cor="#4c78a8", erro=None):
    serie = serie.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 0.32 * len(serie) + 1.2))
    ax.barh(serie.index, serie.values, color=cor, xerr=None if erro is None else erro.iloc[::-1].values, error_kw={"lw": 0.8})
    ax.set_xlabel(xlabel)
    ax.set_title(titulo, fontsize=10)
    return salvar(fig, nome)


def salvar_json(obj, nome: str):
    path = config.TABLES_DIR / nome
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return path


# (a função main, que gera o relatório de avaliação, está no fim do arquivo)


# ---------------------------------------------------------------------------
# Análise exploratória (feita só com 2017-2024: o ano de teste fica intocado)
# ---------------------------------------------------------------------------

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    from scipy.stats import chi2_contingency

    t = pd.crosstab(x.fillna("ausente"), y)
    chi2 = chi2_contingency(t, correction=False)[0]
    n = t.to_numpy().sum()
    r, k = t.shape
    return float(np.sqrt((chi2 / n) / max(min(r - 1, k - 1), 1)))


def figuras_eda(base: pd.DataFrame) -> dict:
    """Gera as figuras e tabelas da EDA. Retorna um dicionário com as tabelas."""
    eda = base[base.ano < config.ANO_TESTE].copy()
    eda["fatal"] = (eda[config.TARGET] == 2).astype(int)
    saidas = {}

    # 1. Viés de seleção: % sem vítimas por ano (2007-2025)
    hist = pd.read_csv(config.EXTERNAL_DIR / "historico_classes_2007_2025.csv").sort_values("ano")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    tot = hist[config.CLASSES].sum(axis=1)
    for c, cor, nome in zip(config.CLASSES, ["#9aa0a6", "#4c78a8", "#e15759"], config.CLASSES_CURTAS):
        ax.plot(hist.ano, 100 * hist[c] / tot, marker="o", ms=3, color=cor, label=nome)
    ax.axvspan(2014.5, 2018.5, color="#f2cf5b", alpha=0.25, lw=0)
    ax.text(2016.5, 95, "declaração online de acidentes\nsem vítimas (jun/2015): transição", fontsize=7.5, va="top", ha="center")
    ax.set_ylabel("% dos acidentes registrados")
    ax.set_title("Acidentes sem vítimas saem da base após 2015;\na composição das classes só se estabiliza em 2019", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.set_ylim(0, 100)
    ax.set_xticks(range(2007, 2026, 2))
    salvar(fig, "eda_01_classes_por_ano.png")
    saidas["historico"] = hist

    # 2. Taxa de acidentes fatais por tipo de acidente
    t = eda.groupby("tipo_acidente").agg(n=("fatal", "size"), pct_fatais=("fatal", "mean")).query("n >= 1000")
    t["pct_fatais"] *= 100
    t = t.sort_values("pct_fatais", ascending=False)
    fig_barras_horizontais(t["pct_fatais"], "% de acidentes fatais por tipo de acidente (2017-2024)", "% fatais",
                           "eda_02_fatais_por_tipo.png", cor="#e15759")
    saidas["por_tipo"] = t

    # 3. Taxa de fatais por hora do dia x tipo de pista
    eda["hora_int"] = eda["hora"].astype(int)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for pista, cor in [("Simples", "#e15759"), ("Dupla", "#4c78a8"), ("Múltipla", "#59a14f")]:
        s = eda[eda.tipo_pista == pista].groupby("hora_int")["fatal"].mean() * 100
        ax.plot(s.index, s.values, marker="o", ms=3, color=cor, label=f"Pista {pista.lower()}")
    ax.set_xlabel("Hora do dia")
    ax.set_ylabel("% fatais")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("Madrugada e pista simples concentram a letalidade", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    salvar(fig, "eda_03_fatais_hora_pista.png")

    # 4. Taxas por variável de contexto
    linhas = []
    for col in ["fase_dia", "tipo_pista", "uso_solo", "condicao_meteo", "regiao", "dia_semana", "faixa_horaria"]:
        g = eda.groupby(col)["fatal"].agg(["size", "mean"]).reset_index()
        for _, r in g.iterrows():
            linhas.append({"variavel": col, "categoria": r[col], "n": int(r["size"]), "pct_fatais": 100 * r["mean"]})
    saidas["taxas_contexto"] = pd.DataFrame(linhas)
    saidas["taxas_contexto"].to_csv(config.TABLES_DIR / "eda_taxas_fatais.csv", index=False)

    # 5. Associação de cada variável com a gravidade (V de Cramér)
    eda["veiculos_cat"] = eda["veiculos"].clip(upper=4).astype(str)
    eda["pessoas_cat"] = eda["pessoas"].clip(upper=6).astype(str)
    cand = ["tipo_acidente", "causa_acidente", "pessoas_cat", "veiculos_cat", "fase_dia", "tipo_pista", "uso_solo",
            "faixa_horaria", "regiao", "uf", "condicao_meteo", "dia_semana", "sentido_via"]
    amostra = eda.sample(min(len(eda), 200_000), random_state=config.SEED)
    v = pd.Series({c: cramers_v(amostra[c], amostra[config.TARGET]) for c in cand}).sort_values(ascending=False)
    cores = ["#e15759" if c in ("causa_acidente", "pessoas_cat") else "#4c78a8" for c in v.index[::-1]]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.barh(v.index[::-1], v.values[::-1], color=cores)
    ax.set_xlabel("V de Cramér com a gravidade (0 = nenhuma associação)")
    ax.set_title("Tipo de acidente é a variável mais associada à gravidade\n(vermelho = só conhecida após o atendimento)", fontsize=10)
    salvar(fig, "eda_04_cramers_v.png")
    saidas["cramers_v"] = v
    v.rename("cramers_v").to_csv(config.TABLES_DIR / "eda_cramers_v.csv")

    # 6. Mapa: letalidade por trecho de 10 km (trechos com >= 30 acidentes)
    tr = eda.groupby("trecho").agg(n=("fatal", "size"), pct=("fatal", "mean"), lat=("latitude", "median"), lon=("longitude", "median"))
    tr = tr[tr.n >= 30].dropna()
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    sc = ax.scatter(tr.lon, tr.lat, c=100 * tr.pct, s=np.clip(tr.n / 15, 2, 40), cmap="YlOrRd", vmin=0, vmax=25, alpha=0.8, lw=0)
    fig.colorbar(sc, ax=ax, fraction=0.04, label="% fatais no trecho")
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("% de acidentes fatais por trecho de 10 km (2017-2024)\ntamanho = nº de acidentes", fontsize=10)
    salvar(fig, "eda_05_mapa_trechos.png")
    return saidas


# ---------------------------------------------------------------------------
# Relatório de avaliação (depois de src.train)
# ---------------------------------------------------------------------------

def _tabela_md(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cab = "| " + " | ".join(df.columns) + " |\n| " + " | ".join("---" for _ in df.columns) + " |\n"
    linhas = []
    for r in df.astype(object).itertuples(index=False):
        cel = []
        for c, v in zip(df.columns, r):
            if c in fmt and pd.notna(v) and not isinstance(v, str):
                cel.append(fmt[c].format(v).replace(",", "."))
            elif isinstance(v, float):
                cel.append("—" if np.isnan(v) else f"{v:.3f}")
            else:
                cel.append(str(v))
        linhas.append("| " + " | ".join(cel) + " |")
    return cab + "\n".join(linhas) + "\n"


def resultados_teste() -> tuple[pd.DataFrame, dict]:
    from src.train import PRED_TESTE

    pred = pd.read_parquet(PRED_TESTE)
    y = pred["y"].to_numpy()
    chaves = sorted({c.split("__")[0] for c in pred.columns if "__" in c})
    linhas, probas, ics = [], {}, {}
    for k in chaves:
        proba = pred[[f"{k}__p{j}" for j in range(3)]].to_numpy()
        p = pred[f"{k}__pred"].to_numpy()
        m = metricas(y, p, proba)
        m["f1_macro_argmax"] = f1_score(y, pred[f"{k}__pred_argmax"], average="macro")
        modelo, cen = k.split("_", 1)
        linhas.append({"modelo": modelo, "cenario": cen, **m})
        probas[k] = proba
        if not k.startswith("dummy"):
            ics[k] = bootstrap_ic(y, p, proba)
    df = pd.DataFrame(linhas)
    for k, ic in ics.items():
        modelo, cen = k.split("_", 1)
        i = df.index[(df.modelo == modelo) & (df.cenario == cen)][0]
        for nome, (lo, hi) in ic.items():
            df.loc[i, nome.replace("_ic95", "_ic_inf")] = lo
            df.loc[i, nome.replace("_ic95", "_ic_sup")] = hi
    df.to_csv(config.TABLES_DIR / "resultados_teste_2025.csv", index=False)
    return df, probas


def importancia_permutacao(base: pd.DataFrame, nome="xgb", cenario="A", n=20_000, repeticoes=5) -> pd.DataFrame:
    """Queda da PR-AUC de Fatais ao embaralhar cada variável de entrada (teste 2025)."""
    import joblib
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import make_scorer

    art = joblib.load(config.MODELS_DIR / f"{nome}_{cenario}.joblib")
    te = base[base.ano == config.ANO_TESTE].sample(n, random_state=config.SEED)
    X, y = te[art["features"]], te[config.TARGET].to_numpy()

    def ap_fatal(y_true, proba):
        return average_precision_score((y_true == 2).astype(int), proba[:, 2])

    scorer = make_scorer(ap_fatal, response_method="predict_proba")
    r = permutation_importance(art["pipeline"], X, y, scoring=scorer, n_repeats=repeticoes, random_state=config.SEED, n_jobs=1)
    df = pd.DataFrame({"variavel": X.columns, "queda_pr_auc_fatais": r.importances_mean, "desvio": r.importances_std})
    df = df.sort_values("queda_pr_auc_fatais", ascending=False)
    df.to_csv(config.TABLES_DIR / f"importancia_permutacao_{nome}_{cenario}.csv", index=False)
    top = df.head(15).set_index("variavel")
    fig_barras_horizontais(top["queda_pr_auc_fatais"], f"Importância por permutação — {NOMES[nome]}, cenário {cenario}\n(queda da PR-AUC de Fatais no teste 2025)",
                           "Queda média da PR-AUC (5 repetições)", f"interp_01_permutacao_{nome}_{cenario}.png", cor=CORES[nome], erro=top["desvio"])
    return df


def shap_xgb(base: pd.DataFrame, cenario="A", n=3000) -> pd.DataFrame:
    """SHAP (TreeExplainer) para a classe Fatais do XGBoost."""
    import joblib
    import shap

    from src.feature_engineering import nomes_legiveis

    art = joblib.load(config.MODELS_DIR / f"xgb_{cenario}.joblib")
    pre, clf = art["pipeline"].named_steps["pre"], art["pipeline"].named_steps["clf"]
    te = base[base.ano == config.ANO_TESTE].sample(n, random_state=config.SEED)
    Xt = pre.transform(te[art["features"]])
    nomes = nomes_legiveis(pre.get_feature_names_out())
    sv = shap.TreeExplainer(clf).shap_values(Xt)
    sv = np.asarray(sv)
    sv_fatal = sv[:, :, 2] if sv.ndim == 3 and sv.shape[-1] == 3 else sv[2]
    imp = pd.DataFrame({"variavel": nomes, "media_abs_shap_fatais": np.abs(sv_fatal).mean(0)}).sort_values("media_abs_shap_fatais", ascending=False)
    imp.to_csv(config.TABLES_DIR / f"shap_xgb_{cenario}.csv", index=False)
    plt.figure()
    shap.summary_plot(sv_fatal, Xt, feature_names=nomes, max_display=15, show=False, plot_size=(7, 5.2))
    plt.title(f"SHAP — classe Fatais (XGBoost, cenário {cenario}, 3.000 acidentes de 2025)", fontsize=10)
    plt.xlabel("Valor SHAP (efeito na log-odds da classe Fatais)")
    salvar(plt.gcf(), f"interp_02_shap_xgb_{cenario}.png")
    return imp


def coeficientes_lr(cenario="A", top=12) -> pd.DataFrame:
    """Coeficientes da Regressão Logística para a classe Fatais (variáveis padronizadas)."""
    import joblib

    from src.feature_engineering import nomes_legiveis

    art = joblib.load(config.MODELS_DIR / f"lr_{cenario}.joblib")
    pre, clf = art["pipeline"].named_steps["pre"], art["pipeline"].named_steps["clf"]
    nomes = nomes_legiveis(pre.get_feature_names_out())
    coef = clf.coef_[2] - clf.coef_.mean(axis=0)  # contraste da classe Fatais com a média das classes
    df = pd.DataFrame({"variavel": nomes, "coef_fatais": coef, "razao_chances": np.exp(coef)})
    df = df.reindex(df.coef_fatais.abs().sort_values(ascending=False).index)
    df.to_csv(config.TABLES_DIR / f"coeficientes_lr_{cenario}.csv", index=False)
    return df.head(top)


def comparacao_regra_simples(pred: pd.DataFrame, chave: str) -> dict:
    """Compara o modelo com a regra "fatal se colisão frontal ou atropelamento de pedestre"."""
    y = pred["y"].to_numpy()
    regra = pred["tipo_acidente"].isin(["Colisão frontal", "Atropelamento de Pedestre"]).to_numpy()
    modelo = pred[f"{chave}__pred"].to_numpy() == 2
    fat = y == 2

    def pr(m, mask=None):
        mask = np.ones_like(fat) if mask is None else mask
        tp = (m & fat & mask).sum()
        prev = (m & mask).sum()
        real = (fat & mask).sum()
        return {"precisao": float(tp / max(prev, 1)), "recall": float(tp / max(real, 1)),
                "f1": float(2 * tp / max(prev + real, 1)), "fatais_reais": int(real), "previstos_fatais": int(prev), "acertos": int(tp)}

    out = {"regra_frontal_ou_pedestre": pr(regra), "modelo": pr(modelo),
           "modelo_dentro_frontal_pedestre": pr(modelo, regra), "modelo_outros_tipos": pr(modelo, ~regra)}
    salvar_json(out, "comparacao_regra_simples.json")
    return out


def gerar_relatorio(base: pd.DataFrame | None = None) -> str:
    """Gera todas as tabelas/figuras de resultados e o arquivo reports/resultados.md."""
    from src.data_processing import carregar_base
    from src.train import PRED_TESTE

    base = carregar_base() if base is None else base
    figuras_eda(base)
    res, probas = resultados_teste()
    pred = pd.read_parquet(PRED_TESTE)
    y = pred["y"].to_numpy()

    # Melhor modelo no cenário A (F1-macro no teste é REPORTADO; a escolha já foi feita na validação)
    val = json.loads((config.TABLES_DIR / "resumo_validacao.json").read_text())
    cand = {k: v["validacao_2024"]["f1_macro_ajustado"] for k, v in val.items() if k.endswith("_A") and not k.startswith("dummy")}
    melhor = max(cand, key=cand.get)
    segundo = sorted(cand, key=cand.get, reverse=True)[1]
    mc = mcnemar(y, pred[f"{melhor}__pred"].to_numpy(), pred[f"{segundo}__pred"].to_numpy())
    mc_lr = mcnemar(y, pred[f"{melhor}__pred"].to_numpy(), pred["lr_A__pred"].to_numpy())
    salvar_json({"melhor_validacao": melhor, "segundo": segundo, "mcnemar_melhor_vs_segundo": mc,
                 "mcnemar_melhor_vs_lr": mc_lr, "f1_validacao": cand}, "comparacao_estatistica.json")

    # Figuras de desempenho
    fig_curvas_pr(y, {k.split("_")[0]: probas[k] for k in probas if k.endswith("_A") and k.count("_") == 1},
                  "res_01_curvas_pr_cenario_A.png", "Curva precisão-recall da classe Fatais — cenário A (teste 2025)")
    fig_calibracao(y, {k.split("_")[0]: probas[k] for k in probas if k.endswith("_A") and k.count("_") == 1}, "res_03_calibracao_A.png")
    fig_matriz_confusao(y, pred[f"{melhor}__pred"].to_numpy(), f"{NOMES[melhor.split('_')[0]]} — cenário A (teste 2025)", "res_02_matriz_confusao_melhor_A.png")
    fig_matriz_confusao(y, pred["lr_A__pred"].to_numpy(), "Regressão Logística — cenário A (teste 2025)", "res_02b_matriz_confusao_lr_A.png")

    # Comparação F1-macro com IC por modelo e cenário
    r = res[res.cenario.isin(["A", "B"])].copy()
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    mods = ["dummy", "lr", "rf", "xgb"]
    for i, cen in enumerate(["B", "A"]):
        for j, m in enumerate(mods):
            row = r[(r.modelo == m) & (r.cenario == cen)]
            if row.empty:
                continue
            row = row.iloc[0]
            x = j + (i - 0.5) * 0.36
            err = None
            if pd.notna(row.get("f1_macro_ic_inf")):
                err = [[row.f1_macro - row.f1_macro_ic_inf], [row.f1_macro_ic_sup - row.f1_macro]]
            ax.bar(x, row.f1_macro, width=0.34, color=CORES[m], alpha=1.0 if cen == "A" else 0.45, yerr=err, error_kw={"lw": 0.8})
            ax.text(x, row.f1_macro + 0.012, f"{row.f1_macro:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(4), [NOMES[m].replace(" (estratificado)", "") for m in mods], fontsize=8.5)
    ax.set_ylabel("F1-macro (teste 2025)")
    ax.set_ylim(0, 0.75)
    ax.set_title("F1-macro por modelo — barras claras: cenário B (só contexto); escuras: cenário A", fontsize=9.5)
    salvar(fig, "res_04_f1_modelos_cenarios.png")

    # Erros por região, mês e tipo de pista (melhor modelo, cenário A)
    pm = pred.assign(pred=pred[f"{melhor}__pred"], p_fatal=pred[f"{melhor}__p2"])
    por_regiao = metricas_por_grupo(pm, "regiao")
    por_uf = metricas_por_grupo(pm, "uf", min_n=800)
    por_mes = metricas_por_grupo(pm, "mes", min_n=100).sort_values("mes")
    por_fase = metricas_por_grupo(pm, "fase_dia")
    por_tipo = metricas_por_grupo(pm, "tipo_acidente", min_n=1000)
    for nome, d in [("regiao", por_regiao), ("uf", por_uf), ("mes", por_mes), ("fase_dia", por_fase), ("tipo_acidente", por_tipo)]:
        d.to_csv(config.TABLES_DIR / f"erros_por_{nome}.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    u = por_uf.sort_values("pr_auc_fatais")
    ax.bar(u["uf"], u["pr_auc_fatais"], color=CORES[melhor.split("_")[0]], label="PR-AUC Fatais")
    ax.plot(u["uf"], u["pct_fatais"] / 100, "k.", label="Prevalência de fatais (referência do acaso)")
    ax.set_ylabel("PR-AUC da classe Fatais")
    ax.set_title(f"Desempenho por UF — {NOMES[melhor.split('_')[0]]}, cenário A (teste 2025)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    salvar(fig, "res_05_desempenho_por_uf.png")

    regra = comparacao_regra_simples(pred, melhor)

    # Interpretação
    perm = importancia_permutacao(base, melhor.split("_")[0], "A")
    shap_imp = shap_xgb(base, "A")
    coef = coeficientes_lr("A")

    # ---------------- Markdown ----------------
    md = ["# Resultados (gerado por `python -m src.evaluate`)\n",
          "Todos os números abaixo são calculados pelos scripts a partir de `reports/predicoes_teste_2025.parquet` e `reports/tables/`.\n"]
    jan = pd.read_csv(config.TABLES_DIR / "janela_treino.csv")
    md.append("## Escolha da janela de treino (validação 2024, XGBoost com parâmetros padrão)\n")
    md.append(_tabela_md(jan[["inicio_treino", "n_treino", "f1_macro_ajustado", "f1_macro_argmax", "pr_auc_fatais", "roc_auc_macro"]],
                         {"n_treino": "{:,.0f}"}))
    cv = pd.read_csv(config.TABLES_DIR / "cv_temporal.csv")
    cvr = cv.groupby("modelo")[["f1_macro_argmax", "pr_auc_fatais", "roc_auc_macro"]].agg(["mean", "std"])
    cvr.columns = [f"{a}_{b}" for a, b in cvr.columns]
    md.append("\n## Validação cruzada temporal (4 folds anuais, 2021-2024, cenário A)\n")
    md.append(_tabela_md(cvr.reset_index()))
    cols = ["modelo", "cenario", "f1_macro", "f1_macro_ic_inf", "f1_macro_ic_sup", "f1_macro_argmax", "recall_fatais", "precision_fatais",
            "f1_fatais", "pr_auc_fatais", "roc_auc_macro_ovr", "acuracia", "recall_fatais_com_precisao_0.30"]
    ordem = {"dummy": 0, "lr": 1, "rf": 2, "xgb": 3}
    tab = res.assign(o=res.modelo.map(ordem)).sort_values(["cenario", "o"])[cols]
    md.append("\n## Teste 2025 (usado uma única vez)\n")
    md.append(_tabela_md(tab))
    md.append(f"\nModelo escolhido na validação (cenário A): **{melhor}**. McNemar contra {segundo}: p = {mc['p_valor']:.3g}; contra lr_A: p = {mc_lr['p_valor']:.3g}.\n")
    sa = json.loads((config.TABLES_DIR / "split_aleatorio.json").read_text())
    md.append(f"\nSplit aleatório (otimista) com XGBoost cenário A: F1-macro (argmax) = {sa['f1_macro_argmax']:.3f}, PR-AUC Fatais = {sa['pr_auc_fatais']:.3f}.\n")
    md.append("\n## Modelo x regra simples (classe Fatais)\n")
    tab_regra = pd.DataFrame(regra).T.reset_index().rename(columns={"index": "comparacao"})
    tab_regra = tab_regra.astype({c: int for c in ("fatais_reais", "previstos_fatais", "acertos")})
    md.append(_tabela_md(tab_regra))
    md.append("\n## Desempenho por região (melhor modelo, cenário A)\n")
    md.append(_tabela_md(por_regiao, {"n": "{:,.0f}", "pct_fatais": "{:.1f}"}))
    md.append("\n## Importância por permutação (top 10)\n")
    md.append(_tabela_md(perm.head(10)))
    md.append("\n## SHAP — classe Fatais (top 10)\n")
    md.append(_tabela_md(shap_imp.head(10)))
    md.append("\n## Regressão Logística — maiores coeficientes da classe Fatais\n")
    md.append(_tabela_md(coef))
    out = config.REPORTS_DIR / "resultados.md"
    out.write_text("\n".join(md), encoding="utf-8")
    return str(out)


def main() -> None:
    print(gerar_relatorio())


if __name__ == "__main__":
    main()
