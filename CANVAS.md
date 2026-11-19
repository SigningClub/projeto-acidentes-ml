# Canvas de abertura do projeto

**Projeto:** Previsão da gravidade de acidentes de trânsito em rodovias federais com Machine Learning, usando dados abertos da Polícia Rodoviária Federal (PRF)
**Disciplina:** Machine Learning — Práticas e Projetos (MBA em Ciência de Dados e IA)
**Data de abertura:** 19/11/2026

O Canvas segue os blocos do *Machine Learning Canvas* (Dorard, 2015). Ele registra as decisões **antes** dos experimentos. Os resultados estão no [README](README.md).

Versão visual: [`docs/canvas.png`](docs/canvas.png).

| Bloco | Conteúdo |
| --- | --- |
| **1. Problema e pergunta** | É possível estimar, no momento do acionamento da PRF, se um acidente terá vítimas fatais, a partir de quando, onde e em que condições ocorreu? Pergunta: modelos de árvores (Random Forest, XGBoost) superam uma Regressão Logística em F1-macro, avaliados num ano futuro (2025) que não entra no treino? |
| **2. Proposta de valor** | Priorizar despacho e resgate nos acidentes com maior chance de serem fatais. Mostrar onde e quando os acidentes tendem a ser mais graves, para orientar fiscalização e obras. |
| **3. Usuários e decisões apoiadas** | Centrais de despacho da PRF: prioridade do atendimento. Gestores da PRF/DNIT e planejadores de segurança viária: escolha de trechos e horários para ações preventivas. É uma ferramenta de apoio; a decisão continua humana. |
| **4. Tarefa de ML** | Classificação supervisionada multiclasse: Sem Vítimas, Com Vítimas Feridas, Com Vítimas Fatais (`classificacao_acidente`). Saída: probabilidade de cada classe. |
| **5. Fonte de dados** | Microdados abertos da PRF, arquivo agrupado por ocorrência (`datatran<ano>.csv`), 2017–2025: 632 mil acidentes, 30 colunas, uma linha por acidente, sem dados pessoais identificáveis. |
| **6. Coleta e rótulos** | O rótulo é oficial: atribuído pelo policial no atendimento a partir do estado das vítimas. Conta só óbitos no local; mortes posteriores aparecem como feridos. |
| **7. Variáveis (sem vazamento)** | Cenário A, principal: tempo (hora, dia, mês, feriado, fase do dia), local (UF, região, BR, km, coordenadas, trecho de 10 km), via (tipo de pista, traçado, uso do solo, sentido), clima, tipo de acidente e nº de veículos. Cenário B: sem tipo de acidente e sem veículos. Cenário C: A + causa do acidente (só para medir vazamento). **Nunca** usadas: mortos, feridos, ilesos, ignorados. |
| **8. Modelos** | Dummy (piso), Regressão Logística (baseline), Random Forest e XGBoost. Desbalanceamento: pesos de amostra e ajuste do limiar de decisão na validação. |
| **9. Avaliação offline** | Divisão temporal: treino 2019–2023, validação 2024, teste 2025 (usado uma vez). Métrica principal: F1-macro. Apoio: PR-AUC da classe Fatais. Também precision, recall e F1 por classe, ROC-AUC, matriz de confusão, IC 95% por bootstrap e McNemar. |
| **10. Uso das previsões** | Probabilidade de acidente fatal por ocorrência (triagem) e mapas de risco por trecho e horário (planejamento). |
| **11. Monitoramento** | Recalcular as métricas a cada ano publicado pela PRF, acompanhar mudanças na distribuição das variáveis e retreinar anualmente. |
| **12. Riscos e ética** | Vazamento por variáveis pós-acidente; viés de seleção (acidentes sem vítimas podem ser declarados online desde 2015 e quase não aparecem na base); viés geográfico; ler importância como causalidade; usar o modelo para negar atendimento. Mitigações: lista de variáveis por cenário com teste automatizado, split temporal, análise por UF/região e linguagem de associação, não de causa. |

## Escopo

| Disciplina (entregue) | TCC (evolução) |
| --- | --- |
| Dados da PRF 2017–2025 | + clima horário (INMET), características da via (DNIT), volume de tráfego |
| Gravidade de um acidente já ocorrido | + risco por trecho e horário (modelos espaço-temporais) |
| Notebooks, módulos, relatório e pitch | + painel interativo e protótipo de apoio à decisão |
