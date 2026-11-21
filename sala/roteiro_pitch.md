# Roteiro do pitch gravado (≈ 19 minutos)

Slides: [Pitch — Previsão da Gravidade de Acidentes (PRF)](https://claude.ai/artifact/NRDx4Rq8DkYP8fcKxqqxhQ). O texto de cada slide abaixo é o mesmo das notas do apresentador.

- **Formato:** gravação de tela (Meet, Teams ou OBS) com os slides em modo apresentação. Durante o slide 10, mostrem o repositório.
- **Divisão:** 3 integrantes. Com 4, o Integrante 4 assume os slides 12–14, e o Integrante 3 fica com 15–17.
- **Antes de gravar:** troquem os marcadores **[Integrante N]** e **[link do repositório no GitHub]** nos slides 1 e 20.
- **Números:** todos estão no `README.md` e em `reports/resultados.md`. Falem como estão escritos aqui.

| Slide | Título | Quem | Tempo |
| --- | --- | --- | --- |
| 1 | Previsão da gravidade de acidentes em rodovias federais | Integrante 1 | ~40 s |
| 2 | Dá para prever, no acionamento da PRF, se um acidente será fatal? | Integrante 1 | ~1 min |
| 3 | Canvas de abertura do projeto | Integrante 1 | ~1 min |
| 4 | 632 mil acidentes, um por linha, sem dados pessoais | Integrante 1 | ~50 s |
| 5 | Acidentes sem vítimas saíram da base depois de 2015 | Integrante 1 | ~1 min 10 s |
| 6 | Frontal e atropelamento: 28% fatais | Integrante 2 | ~50 s |
| 7 | Hora, via e região também importam | Integrante 2 | ~1 min |
| 8 | Só entra o que se sabe no acionamento | Integrante 2 | ~1 min 10 s |
| 9 | Treino no passado, teste no futuro | Integrante 2 | ~1 min 10 s |
| 10 | Cinco módulos, seis notebooks e testes | Integrante 2 | ~1 min 20 s |
| 11 | Três modelos e uma métrica principal | Integrante 2 | ~1 min |
| 12 | 0,561 — F1-macro do XGBoost | Integrante 3 | ~30 s |
| 13 | XGBoost lidera, por margem pequena | Integrante 3 | ~1 min |
| 14 | Acha 42% dos fatais, com 39% de precisão | Integrante 3 | ~1 min |
| 15 | O tipo de acidente faz a diferença | Integrante 3 | ~1 min |
| 16 | O limite: fora de frontal e atropelamento | Integrante 3 | ~1 min 10 s |
| 17 | Varia por UF; probabilidades a recalibrar | Integrante 3 | ~1 min |
| 18 | O que concluímos | Integrante 1 | ~50 s |
| 19 | Limitações e ponte para o TCC | Integrante 1 | ~50 s |
| 20 | Perguntas? | Integrante 1 | ~20 s |

Tempos por integrante: Integrante 1 ≈ 6 min 40 s (slides 1–5 e 18–20) · Integrante 2 ≈ 6 min 30 s (6–11) · Integrante 3 ≈ 5 min 40 s (12–17).

---

### Slide 1 — Previsão da gravidade de acidentes em rodovias federais

**Integrante 1 · ~40 s**

Olá, professor. Somos [nomes do grupo] e vamos apresentar o projeto final da disciplina Machine Learning — Práticas e Projetos.

O tema é a previsão da gravidade de acidentes de trânsito nas rodovias federais, usando os dados abertos da Polícia Rodoviária Federal.

Eu vou falar do problema e dos dados; [Integrante 2] apresenta a análise exploratória, a metodologia e o código; e [Integrante 3] mostra os resultados e a interpretação. No fim eu volto para as conclusões e a ponte para o TCC.

Todo o código, os notebooks, o Canvas e o README com os resultados estão no repositório, cujo link aparece no último slide.

### Slide 2 — Dá para prever, no acionamento da PRF, se um acidente será fatal?

**Integrante 1 · ~1 min**

As rodovias federais registram cerca de 70 mil acidentes por ano atendidos pela PRF. Em 2025 foram 72.529.

A PRF classifica cada acidente em três classes: sem vítimas, com vítimas feridas e com vítimas fatais. Em 2025, 7,2% foram fatais: 5.209 acidentes.

Nossa pergunta é: com o que se sabe no momento em que a PRF é acionada — quando, onde, em que via, com que clima, que tipo de acidente e quantos veículos — dá para prever se o acidente será fatal?

É uma classificação multiclasse, com foco na classe mais rara e mais grave.

A aplicação é apoio à decisão: priorizar despacho e resgate e orientar fiscalização e obras. O modelo não serve para negar ou reduzir atendimento a ninguém.

### Slide 3 — Canvas de abertura do projeto

**Integrante 1 · ~1 min**

Antes de qualquer código, fizemos o Canvas de abertura, que está no repositório como CANVAS.md e em imagem.

Destaco quatro blocos:

— Proposta de valor: priorizar despacho e resgate nos casos mais graves e mostrar onde e quando a gravidade é maior.

— Variáveis: separamos o que é conhecido no acionamento do que só existe depois do atendimento. Isso vai aparecer na metodologia.

— Avaliação: treino 2019–2023, validação 2024 e teste 2025, com F1-macro como métrica principal.

— Riscos: vazamento, viés de seleção, viés geográfico, leitura causal indevida e uso para negar atendimento.

### Slide 4 — 632 mil acidentes, um por linha, sem dados pessoais

**Integrante 1 · ~50 s**

Os dados são os microdados abertos da PRF, no arquivo “agrupados por ocorrência”: uma linha por acidente, 30 colunas e nenhum dado pessoal identificável.

Usamos de 2017 a 2025: 632.420 acidentes.

As colunas cobrem tempo, local, via, clima, tipo e causa do acidente e as contagens de pessoas, veículos e vítimas.

Sobre a procedência: o código baixa da página oficial da PRF. No ambiente em que rodamos, o Google Drive da PRF estava bloqueado, então usamos cópias públicas dos mesmos arquivos. O hash de cada um está registrado no manifest.json, e os de 2023 a 2025 conferem entre repositórios independentes.

### Slide 5 — Acidentes sem vítimas saíram da base depois de 2015

**Integrante 1 · ~1 min 10 s**

Este é o achado mais importante sobre a base.

Até 2014, cerca de 60% dos acidentes registrados não tinham vítimas. Em junho de 2015, a PRF passou a aceitar a declaração online de acidentes sem vítimas, sem atendimento policial. Esses acidentes deixaram a base.

O gráfico mostra a queda: a fração de sem vítimas cai até 2018 e só se estabiliza, entre 15% e 18%, a partir de 2019.

Duas consequências para o projeto:

Primeiro, o modelo vale para acidentes atendidos pela PRF, não para todos os acidentes.

Segundo, começamos o treino em 2019. E não foi só intuição: testamos 2017, 2019 e 2021 como início na validação, e 2019 teve o melhor F1-macro.

Agora o [Integrante 2] mostra a análise exploratória.

### Slide 6 — Frontal e atropelamento: 28% fatais

**Integrante 2 · ~50 s**

Na análise exploratória usamos só 2017 a 2024. O ano de 2025, que é o teste, ficou de fora, para nenhuma decisão ser tomada olhando para ele.

O tipo de acidente é o fator mais marcante: colisões frontais e atropelamentos de pedestre são fatais em cerca de 28% dos casos; colisões traseiras, em 4%.

Medimos a associação de cada variável com a gravidade pelo V de Cramér, e o tipo de acidente ficou em primeiro, com 0,32.

Lembrando: isso é associação. O tipo descreve a dinâmica do acidente, não a sua causa.

### Slide 7 — Hora, via e região também importam

**Integrante 2 · ~1 min**

O contexto também importa. O gráfico mostra a porcentagem de acidentes fatais por hora do dia em cada tipo de pista: a pista simples, em vermelho, tem a maior letalidade, com pico na madrugada.

Em área não urbana, 8,6% dos acidentes são fatais, contra 4,4% em área urbana. Nordeste e Norte têm as maiores taxas; Sul e Sudeste, as menores.

Um dado curioso: com chuva a letalidade é menor que com céu claro. Uma explicação plausível é que a chuva aumenta os acidentes leves e reduz a velocidade. É um bom exemplo de que taxa descritiva não é efeito causal.

Mais um ponto da EDA: causa do acidente e número de pessoas também têm associação forte, mas só são conhecidos depois do atendimento. Por isso ficam fora do modelo principal, como vou mostrar agora.

### Slide 8 — Só entra o que se sabe no acionamento

**Integrante 2 · ~1 min 10 s**

O cuidado central do projeto é o vazamento de dados.

A classe do acidente é calculada a partir das contagens: se há um morto, é fatal; se há feridos, é com feridos. Então mortos, feridos e ilesos nunca podem entrar no modelo.

E há variáveis mais sutis: a causa do acidente é presumida pelo policial depois do levantamento. Ela não existe quando a chamada chega.

Por isso definimos cenários de informação, todos em src/config.py:

— Cenário A, o principal: o que se sabe no acionamento, incluindo tipo de acidente e número de veículos.

— Cenário B: só o contexto — tempo, local, via e clima —, útil para planejamento.

— Cenário C: acrescenta a causa. Não é previsão; serve só para medir quanto a informação pós-acidente infla as métricas.

E temos um teste automatizado que falha se uma coluna de desfecho chegar ao modelo.

### Slide 9 — Treino no passado, teste no futuro

**Integrante 2 · ~1 min 10 s**

A divisão dos dados é temporal: o modelo aprende com o passado e é avaliado num ano que nunca viu.

Treino de 2019 a 2023, com 327 mil acidentes; validação em 2024; e teste em 2025.

O protocolo tem quatro etapas:

1. Escolhemos o ano inicial do treino olhando só a validação. 2019 venceu.

2. Fizemos busca aleatória de hiperparâmetros, tratando os pesos de classe — balanceado, raiz ou nenhum — como mais um hiperparâmetro, para lidar com o desbalanceamento.

3. Ajustamos o limiar de decisão em 2024: multiplicamos as probabilidades das classes raras por pesos que maximizam o F1-macro.

4. Retreinamos em 2019 a 2024 e prevemos 2025 uma única vez.

Além disso, validação cruzada temporal com quatro folds anuais, intervalos de confiança por bootstrap e teste de McNemar entre os modelos.

### Slide 10 — Cinco módulos, seis notebooks e testes

**Integrante 2 · ~1 min 20 s** · se quiserem, mostrem o repositório na tela durante este slide

O código está organizado em cinco módulos em src/, mais os testes.

— config.py concentra tudo o que define o experimento: anos de corte, variáveis de cada cenário e colunas proibidas.

— data_processing.py baixa os arquivos, lê com a codificação correta, harmoniza categorias que mudaram de nome ao longo dos anos e cria atributos como hora em seno e cosseno, feriados e o trecho de 10 km.

— feature_engineering.py tem o pré-processamento dentro de Pipelines do scikit-learn: One-Hot e padronização para a regressão; codificação ordinal para as árvores; target encoding do trecho com cross fitting, para não vazar o alvo; e o ajuste de limiar.

— train.py roda o protocolo inteiro, e evaluate.py gera métricas, figuras e a interpretação.

Os seis notebooks seguem as etapas pedidas e importam src/, sem duplicar código. Os 11 testes passam.

### Slide 11 — Três modelos e uma métrica principal

**Integrante 2 · ~1 min**

Comparamos três modelos, além de um classificador aleatório como piso:

— Regressão Logística, o baseline: linear, interpretável, padrão na literatura de segurança viária, mas só capta interações que a gente cria à mão.

— Random Forest: muitas árvores profundas treinadas em paralelo; a média dos votos reduz a variância.

— XGBoost: árvores rasas em sequência, cada uma corrigindo o erro das anteriores, com regularização. Costuma ser o melhor em dados tabulares.

A métrica principal, definida antes dos experimentos, é o F1-macro. Ela dá o mesmo peso às três classes. A acurácia enganaria: prever sempre “feridas” daria 77% de acurácia sem encontrar nenhum acidente fatal.

Como apoio, usamos a PR-AUC da classe Fatais, que mede o ranqueamento sem depender de limiar.

Agora o [Integrante 3] apresenta os resultados.

### Slide 12 — 0,561 — F1-macro do XGBoost

**Integrante 3 · ~30 s**

O resultado principal: no teste com os 72.529 acidentes de 2025, o XGBoost alcançou F1-macro de 0,561, com intervalo de confiança de 95% entre 0,555 e 0,566.

Para comparação, a Regressão Logística teve 0,543 e o classificador aleatório, 0,334.

E um detalhe que mostra estabilidade: na validação de 2024 o XGBoost também teve 0,561.

### Slide 13 — XGBoost lidera, por margem pequena

**Integrante 3 · ~1 min**

Aqui estão todos os modelos no cenário A, no teste de 2025.

O XGBoost é o melhor em todas as métricas: F1-macro de 0,561, recall de 42% e precisão de 39% na classe fatal, e PR-AUC de 0,348.

O Random Forest vem em seguida, com 0,551, e a Regressão Logística, com 0,543.

O teste de McNemar mostra que as diferenças são estatisticamente significativas, mas precisamos ser honestos: o ganho sobre o baseline é pequeno, 1,8 ponto de F1-macro. A Regressão Logística continua competitiva e é mais interpretável.

### Slide 14 — Acha 42% dos fatais, com 39% de precisão

**Integrante 3 · ~1 min**

Olhando para a classe que mais importa, a dos acidentes fatais:

À esquerda, a curva precisão-recall. O XGBoost tem PR-AUC de 0,348, quase cinco vezes a do acaso, que é a linha tracejada em 0,072. Se o gestor aceitar uma precisão de 30%, o modelo encontra 55% dos acidentes fatais.

À direita, a matriz de confusão normalizada. A classe com feridos é bem reconhecida, com recall de 86%. A dificuldade está nas fronteiras: 51% dos acidentes fatais e 59% dos sem vítimas são previstos como “feridas”, que é a classe majoritária e mais heterogênea.

### Slide 15 — O tipo de acidente faz a diferença

**Integrante 3 · ~1 min**

Os cenários de informação mostram de onde vem o desempenho.

Só com contexto — tempo, local, via e clima —, o cenário B, o F1-macro cai para 0,442. Ainda supera o acaso, então serve para planejamento, mas não para triagem individual.

O salto para 0,561 vem de acrescentar o tipo de acidente e o número de veículos.

Com a causa do acidente, o cenário C, o F1 sobe só para 0,576. O vazamento foi pequeno aqui, mas esse modelo não serve para prever, porque a causa não existe no acionamento.

E testamos um split aleatório: o resultado foi praticamente igual ao temporal. Nesta base ele não inflou as métricas, mas o temporal continua sendo o protocolo correto, porque reproduz o uso real.

### Slide 16 — O limite: fora de frontal e atropelamento

**Integrante 3 · ~1 min 10 s**

O que o modelo aprendeu? A importância por permutação mostra um fator dominante: embaralhar o tipo de acidente derruba a PR-AUC de Fatais em 0,200. Depois vêm o número de veículos, o trecho da rodovia, a latitude, a hora e o uso do solo. O SHAP, que está no notebook 06, confirma.

Isso nos levou a uma comparação honesta: uma regra simples — “é fatal se for colisão frontal ou atropelamento” — teria F1 de 0,353 na classe fatal. O modelo tem 0,405, então vai além da regra.

Mas nesses dois tipos o modelo encontra 85% dos fatais; nos demais tipos, que somam mais da metade dos acidentes fatais, encontra só 8%. É o principal limite do trabalho: o que distingue um tombamento fatal de um não fatal — velocidade, tipo de veículo, ocupantes — não está nas variáveis do acionamento.

Lembrando sempre: importância explica o modelo, não as causas dos acidentes.

### Slide 17 — Varia por UF; probabilidades a recalibrar

**Integrante 3 · ~1 min**

Investigamos vieses geográficos e temporais.

Entre as UFs, a PR-AUC de Fatais vai de 0,24 no Rio Grande do Norte a 0,53 no Ceará. O recall de Fatais é maior no Norte e no Nordeste, onde os acidentes fatais são mais frequentes, e maior à noite do que de dia.

No tempo, o modelo é estável: o F1-macro foi 0,561 na validação de 2024 e no teste de 2025, e a validação cruzada temporal teve desvio pequeno.

Um ponto importante para o uso prático: como treinamos as árvores com pesos de classe, as probabilidades do XGBoost superestimam a chance de acidente fatal. Para ranquear e classificar isso não atrapalha, porque o limiar foi ajustado, mas antes de comunicar o número como “risco” é preciso recalibrar. A Regressão Logística, treinada sem pesos, está bem calibrada.

Passo para o [Integrante 1] com as conclusões.

### Slide 18 — O que concluímos

**Integrante 1 · ~50 s**

Para concluir, quatro pontos:

Primeiro: é possível prever a gravidade bem acima do acaso, usando só o que se sabe no acionamento. F1-macro de 0,561 e PR-AUC da classe fatal quase cinco vezes o acaso, num ano futuro.

Segundo: respondendo à pergunta de pesquisa, as árvores superam a Regressão Logística, mas por pouco. A regressão continua competitiva, mais interpretável e mais bem calibrada.

Terceiro: o desempenho vem sobretudo do tipo de acidente. Só com contexto, o modelo serve para planejamento, não para triagem de cada ocorrência.

Quarto, do ponto de vista metodológico: separar cenários de informação e testar num ano futuro nos permitiu medir o vazamento e o viés de seleção em vez de supor.

### Slide 19 — Limitações e ponte para o TCC

**Integrante 1 · ~50 s**

As principais limitações:

— A base só tem acidentes atendidos pela PRF; os sem vítimas declarados online ficam de fora.

— O rótulo conta óbito só no local. Vítimas que morrem depois aparecem como feridas.

— Clima e traçado são anotados pelo policial, e não temos velocidade, tipo de veículo nem fluxo.

— E os arquivos vieram de cópias públicas dos CSV oficiais, com hash registrado.

Para o TCC, a pergunta é: integrar clima medido do INMET, dados da via do DNIT e os tipos de veículo envolvidos melhora a detecção de acidentes fatais justamente fora das colisões frontais e atropelamentos? E permite apontar trechos críticos estáveis de um ano para o outro?

Junto com isso: recalibrar as probabilidades, modelar o risco por trecho e hora e construir um painel para gestores.

### Slide 20 — Perguntas?

**Integrante 1 · ~20 s** · todos podem aparecer na câmera

Era isso. Todo o projeto — código, os seis notebooks executados, o Canvas e o README com os resultados — está no repositório, cujo link está na tela e também no relatório técnico.

Obrigado, professor.



Tempo total estimado: cerca de 19 minutos (mínimo exigido: 15).

---

### Dicas de gravação

- Ensaiem uma vez com cronômetro: o mínimo exigido é 15 minutos.
- Cada integrante deve aparecer falando (câmera ligada ou voz identificada).
- Se o tempo passar de 20 minutos, encurtem os slides 10 e 15, não os resultados.
