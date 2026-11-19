"""Gera docs/canvas.png (versão visual do CANVAS.md). Uso: python docs/gerar_canvas.py"""
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLOCOS = [
    ("1. Problema", "Estimar, no acionamento da PRF, se um acidente terá vítimas fatais. Árvores superam Regressão Logística em F1-macro num ano futuro (2025)?"),
    ("2. Proposta de valor", "Priorizar despacho e resgate nos acidentes mais graves; mostrar onde e quando a gravidade é maior para orientar prevenção."),
    ("3. Usuários e decisões", "Centrais de despacho (prioridade); gestores PRF/DNIT (trechos e horários para fiscalização e obras). Apoio, não decisão automática."),
    ("4. Tarefa de ML", "Classificação multiclasse: Sem Vítimas, Com Vítimas Feridas, Com Vítimas Fatais. Saída probabilística."),
    ("5. Fonte de dados", "Dados abertos PRF, agrupados por ocorrência, 2017-2025: 632 mil acidentes, 30 colunas, sem dados pessoais."),
    ("6. Rótulos", "Classificação oficial atribuída pelo policial. Óbito só no local; mortes posteriores contam como feridos."),
    ("7. Variáveis", "A (principal): tempo, local, via, clima, tipo de acidente, nº de veículos. B: só contexto. C: A + causa (mede vazamento). Nunca: mortos, feridos, ilesos."),
    ("8. Modelos", "Dummy, Regressão Logística (baseline), Random Forest, XGBoost. Pesos de amostra + ajuste de limiar na validação."),
    ("9. Avaliação offline", "Treino 2019-2023, validação 2024, teste 2025 (uma vez). Principal: F1-macro. Apoio: PR-AUC Fatais, IC bootstrap, McNemar."),
    ("10. Uso das previsões", "Probabilidade de acidente fatal por ocorrência; mapas de risco por trecho e horário."),
    ("11. Monitoramento", "Métricas a cada ano publicado; deriva das variáveis; retreino anual."),
    ("12. Riscos e ética", "Vazamento, viés de seleção (e-DAT), viés geográfico, leitura causal indevida, uso para negar atendimento."),
]

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 4)
ax.set_ylim(0, 3.25)
ax.axis("off")
ax.text(0.02, 3.17, "Canvas de abertura — Previsão da gravidade de acidentes em rodovias federais (dados PRF)",
        fontsize=16, fontweight="bold", va="center")
ax.text(0.02, 3.07, "Machine Learning — Práticas e Projetos · 19/11/2026 · blocos do Machine Learning Canvas",
        fontsize=10.5, color="#555", va="center")
cores = ["#e8f0fb", "#e8f0fb", "#e8f0fb", "#fdecea", "#eef6ec", "#eef6ec", "#eef6ec", "#fdecea", "#fdecea", "#e8f0fb", "#f4f4f4", "#fff4d6"]
for i, (titulo, texto) in enumerate(BLOCOS):
    lin, col = divmod(i, 4)
    x, y = col + 0.03, 2.0 - lin * 1.0 + 0.03
    ax.add_patch(FancyBboxPatch((x, y), 0.94, 0.92, boxstyle="round,pad=0.0,rounding_size=0.03", fc=cores[i], ec="#999", lw=0.8))
    ax.text(x + 0.04, y + 0.85, titulo, fontsize=12, fontweight="bold", va="top")
    ax.text(x + 0.04, y + 0.74, "\n".join(textwrap.wrap(texto, 36)), fontsize=10.2, va="top", linespacing=1.35)
out = Path(__file__).with_name("canvas.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print(out)
