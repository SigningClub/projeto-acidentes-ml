# Dados

Os CSV não são versionados no Git (cerca de 180 MB descompactados). Para reproduzir o projeto, baixe-os da fonte oficial:

```bash
python -m src.data_processing --baixar    # 2017-2025, Google Drive oficial da PRF
python -m src.data_processing             # limpa e grava data/processed/acidentes_2017_2025.parquet
```

## Fonte oficial

- Página: [Dados Abertos da PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf), arquivo **"Documento CSV de Acidentes — Agrupados por ocorrência"** (`datatran<ano>.csv`).
- Dicionário de variáveis: [Dicionário de dados — Acidentes](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dicionario-acidentes).
- Os IDs do Google Drive de cada ano estão em `src/config.py` (`PRF_DRIVE_IDS`). Foram copiados da página oficial em 19/11/2026.
- Formato: Latin-1, separador `;`. O separador decimal de latitude/longitude muda entre anos (vírgula ou ponto); `data_processing._num` aceita os dois.

## Origem dos arquivos usados nos resultados deste repositório

O ambiente onde os experimentos foram executados não tinha acesso ao Google Drive. Por isso, os arquivos foram obtidos de **cópias públicas no GitHub** dos mesmos CSV oficiais. Os hashes SHA-256 estão em `raw/manifest.json`.

| Ano | Acidentes | Espelho |
| --- | --- | --- |
| 2017–2021 | 89.563 / 69.295 / 67.446 / 63.548 / 64.515 | [FelipeGBarreto/analise_cluster_acidentes_rodovias_federais](https://github.com/FelipeGBarreto/analise_cluster_acidentes_rodovias_federais) |
| 2022–2023 | 64.606 / 67.766 | [LeticiaHms/mack-car-crash-prediction](https://github.com/LeticiaHms/mack-car-crash-prediction) |
| 2024 | 73.156 | [Alli-will/content-datatran2024](https://github.com/Alli-will/content-datatran2024) |
| 2025 | 72.529 | [Nickolas3211/datatran2025](https://github.com/Nickolas3211/datatran2025) |

**Verificação feita:**

- Todos os arquivos têm o mesmo cabeçalho de 30 colunas e codificação Latin-1.
- Os arquivos de 2023, 2024 e 2025 foram encontrados em dois ou mais repositórios independentes com conteúdo idêntico (diferem só pela quebra de linha CRLF/LF).
- 2017–2021 vêm de um único espelho. Os totais por ano são compatíveis com as séries publicadas pela PRF, mas recomenda-se rodar `--baixar` e comparar com `raw/manifest.json` antes de uma entrega definitiva.
- O arquivo de 2022 de outro espelho (`diegotsilva94/datatran2022`) foi **descartado**, porque fora alterado (datas reformatadas e 10 mil linhas a menos).

A PRF publica os dados em janeiro e os revisa em março; versões baixadas em datas diferentes podem ter pequenas diferenças.

## Arquivos versionados

- `raw/manifest.json`: tamanho, hash e espelho de cada CSV.
- `external/historico_classes_2007_2025.csv`: contagem de acidentes por classe e ano, de 2007 a 2025 (usada no gráfico do viés de seleção; 2007–2016 do mesmo espelho de 2017–2021).

## Dados pessoais

O arquivo por ocorrência não traz nome, documento ou placa. O projeto não usa os arquivos agrupados por pessoa (idade e sexo).
