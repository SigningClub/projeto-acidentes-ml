# Roteiro do pitch gravado (≈ 17 minutos)

Formato sugerido: gravação de tela (Google Meet, Teams ou OBS), mostrando o **README**, as **figuras** e os **notebooks** do repositório. Cada integrante fala em pelo menos um bloco. A divisão abaixo supõe 3 integrantes (I1, I2, I3); com 4, separe o bloco 5 em dois.

Todos os números citados estão no `README.md` e em `reports/resultados.md`.

| Bloco | Tempo | Quem | O que mostrar na tela |
| --- | --- | --- | --- |
| 1. Problema e motivação | 2 min | I1 | README (topo) e `docs/canvas.png` |
| 2. Dados e viés de seleção | 2,5 min | I1 | `eda_01_classes_por_ano.png`, `data/README.md` |
| 3. EDA | 2,5 min | I2 | notebook 02: figuras de tipo, hora × pista, V de Cramér, mapa |
| 4. Metodologia e código | 4 min | I2 | `src/config.py`, `src/train.py`, notebook 03, `tests/` |
| 5. Resultados | 3,5 min | I3 | README "Resultados em uma página", curva PR, matriz de confusão |
| 6. Interpretação, limites e TCC | 2,5 min | I3 → I1 | SHAP, permutação, tabela da regra simples, seções 7–9 do README |

---

## Bloco 1 — Problema e motivação (I1, 2 min)

- "As rodovias federais têm cerca de 70 mil acidentes por ano atendidos pela PRF. Em 2025 foram 72.529, e 7,2% tiveram vítimas fatais."
- "Nossa pergunta: com o que se sabe no momento em que a PRF é acionada (quando, onde, em que via, com que clima, que tipo de acidente e quantos veículos), dá para prever se o acidente será fatal?"
- "É uma classificação com três classes oficiais: sem vítimas, com feridos e com vítimas fatais. O foco é a classe fatal, a mais rara."
- Mostrar o Canvas: proposta de valor (priorizar despacho e resgate; orientar fiscalização), usuários, riscos.
- "O modelo é apoio à decisão. Ele não serve para negar ou reduzir atendimento."

## Bloco 2 — Dados e viés de seleção (I1, 2,5 min)

- "Usamos os dados abertos da PRF, arquivo agrupado por ocorrência, de 2017 a 2025: 632 mil acidentes e 30 colunas, sem dados pessoais."
- Procedência: "o código baixa da página oficial; neste ambiente usamos cópias públicas dos mesmos arquivos, com hash registrado no `manifest.json`."
- **Achado principal da base** (figura de classes por ano): "até 2014, cerca de 60% dos acidentes eram sem vítimas. Em 2015 a PRF passou a aceitar a declaração online de acidentes sem vítimas, e essa fração caiu para 15–18%. A composição só se estabiliza em 2019."
- "Consequências: (1) o modelo vale para acidentes atendidos pela PRF; (2) começamos o treino em 2019, uma escolha que confirmamos na validação."

## Bloco 3 — Análise exploratória (I2, 2,5 min)

- Tipo de acidente: "colisão frontal e atropelamento de pedestre são fatais em cerca de 28% dos casos; colisão traseira, em 4%."
- Hora × pista: "madrugada e pista simples concentram a letalidade."
- Uso do solo e região: "área não urbana tem 8,6% de fatais contra 4,4% na urbana; Norte e Nordeste têm as maiores proporções."
- V de Cramér: "o tipo de acidente é a variável mais associada. Causa e número de pessoas também aparecem fortes, mas só são conhecidos depois do atendimento. Por isso ficam fora do modelo principal."
- "Chuva tem letalidade menor que céu claro. Isso mostra que taxa descritiva não é causa."

## Bloco 4 — Metodologia e código (I2, 4 min)

- **Vazamento:** mostrar `CENARIOS` em `src/config.py`.
  - "Mortos, feridos e ilesos definem o alvo e nunca entram."
  - "Cenário A, o principal, simula a chamada. B usa só o contexto. C acrescenta a causa e serve só para medir o vazamento."
  - Mostrar o teste `test_cenarios_sem_colunas_de_desfecho`.
- **Divisão temporal:** "treino 2019–2023, validação 2024, teste 2025. O teste foi usado uma única vez."
- **Pipeline** (`feature_engineering.py`):
  - "One-Hot e padronização para a Regressão Logística; codificação ordinal para as árvores."
  - "*Target encoding* do trecho de 10 km com *cross fitting*: cada linha de treino é codificada sem ela mesma."
  - "Tudo dentro do `Pipeline`, ajustado só no treino."
- **Desbalanceamento:** "pesos de classe como hiperparâmetro, mais ajuste do limiar de decisão escolhido na validação para maximizar o F1-macro."
- **Protocolo** (`train.py`): janela → busca de hiperparâmetros → modelo final → validação cruzada temporal → split aleatório para comparação.
- **Métrica:** "F1-macro como principal, porque a acurácia premiaria prever sempre 'feridas'. Como apoio, a PR-AUC da classe fatal, que não depende de limiar."

## Bloco 5 — Resultados (I3, 3,5 min)

- Tabela do README:
  - "XGBoost: F1-macro de 0,561, com intervalo de 0,555 a 0,566."
  - "Random Forest: 0,551. Regressão Logística: 0,543. Dummy: 0,334."
- "A diferença é estatisticamente significativa (McNemar), mas pequena: 1,8 ponto sobre o baseline."
- Curva PR: "o XGBoost encontra 42% dos acidentes fatais com 39% de precisão. A PR-AUC é 0,348, quase 5 vezes o acaso. Aceitando 30% de precisão, encontra 55% dos fatais."
- Cenários:
  - "Só com contexto, cai para 0,44."
  - "Com a causa, sobe para 0,576: o ganho do vazamento foi pequeno, mas esse modelo não serve para prever."
- "O split aleatório deu praticamente o mesmo resultado do temporal. Mesmo assim, o temporal é o protocolo correto."
- Matriz de confusão: "metade dos fatais é prevista como feridos. A classe feridos é a fronteira difícil."

## Bloco 6 — Interpretação, limites e TCC (I3 → I1, 2,5 min)

- I3, permutação e SHAP:
  - "Tipo de acidente domina; depois vêm nº de veículos, trecho, hora e uso do solo. Essas medidas explicam o modelo, não as causas."
  - Tabela da regra simples: "a regra 'fatal se frontal ou atropelamento' teria F1 de 0,353; o modelo tem 0,405. Mas fora desses dois tipos o modelo acha só 8% dos fatais. Esse é o nosso principal limite."
  - Por UF: "o desempenho varia entre UFs: a PR-AUC vai de 0,24 no RN a 0,53 no CE."
  - Calibração: "as probabilidades do XGBoost precisam ser recalibradas antes de virar escore de risco."
- I1, conclusão e TCC:
  - "Dá para prever bem acima do acaso no acionamento, com ganho modesto das árvores sobre a regressão."
  - "No TCC: clima medido do INMET, dados da via do DNIT, tipos de veículo e risco espaço-temporal por trecho, com um painel para gestores."
  - Encerramento: link do repositório.

---

### Dicas de gravação

- Ensaie uma vez com cronômetro: o mínimo exigido é 15 minutos.
- Antes de gravar, abra os arquivos em abas: README, `docs/canvas.png`, notebooks 02, 03 e 06, `src/config.py`.
- Fale os números como estão no README: não arredonde de forma diferente entre os integrantes.
