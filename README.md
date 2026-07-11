# Análise LDA — Corpus de Digitalização Portuária (125 artigos, 2007–2026)

Pipeline completo de topic modeling (Latent Dirichlet Allocation) sobre um corpus
de 125 artigos, para a tese. Metodologia alinhada com Ferdinand et al.
(2024), *Topic Identification of Science and Mathematics Literature Using LDA*.

Os PDFs de origem colocam-se numa pasta `LDA_PAPERS/` na raiz do repositório.
Essa pasta **não é distribuída** (ver [Dados e copyright](#dados-e-copyright)).

## Resultado principal

**k = 12 tópicos** (gensim LdaModel, Variational Bayes, alpha=0.1, eta=0.01,
seed=42, dicionário de 5.002 termos após `no_below=5` / `no_above=0.5`).

| # | Tópico | Artigos | Prevalência |
|---|--------|---------|-------------|
| T10 | Governance and sustainable digital transformation of ports | 22 | 16.5% |
| T8 | Port community systems for import/export documentation flows | 20 | 15.0% |
| T6 | National maritime single window: architecture and implementation | 17 | 11.0% |
| T0 | Port call coordination and inter-organizational information sharing | 11 | 9.6% |
| T7 | Single-window trade facilitation: national implementation cases | 13 | 9.3% |
| T3 | Digital innovation adoption: barriers, stakeholders and maturity | 8 | 9.0% |
| T1 | Cybersecurity and safety of digital maritime infrastructure | 8 | 7.9% |
| T9 | Smart-port evaluation: indicators, KPIs and adoption strategies | 7 | 6.5% |
| T4 | Research trends and bibliometric mapping of port digitalization | 4 | 4.9% |
| T5 | Blockchain for trade documentation and port community systems | 8 | 4.5% |
| T2 | Operational optimization, resilience and service quality | 5 | 4.4% |
| T11 | Digital-platform business models in freight transport | 2 | 1.5% |

Seleção de k: varrimento C_V para k=3–20 (3 seeds cada; Röder et al., 2015).
Patamar de coerência começa em k≈10–12; k=12 tem C_V=0.4920±0.0075 (menor
variância do varrimento; máximo global k=14 estatisticamente indistinguível).
A escolha final entre k∈{10,12,14} foi validada por avaliação estruturada de
interpretabilidade. Seed final (42) escolhida entre 5 inicializações por
interpretabilidade (C_V a menos de 1 dp do máximo).

Estabilidade (limitação a reportar): ARI médio entre seeds = 0.135; overlap
médio dos top-10 termos = 34%.

## Instalação

Testado em Python 3.11.

```
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords
```

## Dados e copyright

O texto integral dos 125 artigos (a maioria atrás de paywall) e a sua versão
tokenizada **não são distribuídos** neste repositório, por razões de copyright:
`data/raw_text/`, `data/tokens.json` e `ponto2_robustez/data_variants/` estão
excluídos via `.gitignore`. São regeneráveis a partir dos PDFs originais.

Para reproduzir de raiz, coloque os PDFs em `LDA_PAPERS/` (na raiz) e corra os
passos 01–02 abaixo, que geram `data/raw_text/` e `data/tokens.json`. Os passos
03–07 dependem desses dados (`data/tokens.json` e/ou `data/raw_text/`), pelo que
**exigem** ter corrido antes o 01–02.

O que é distribuído: todo o código, os resultados agregados (`results/`), o
modelo treinado (`results/final_model/`), as figuras, o `data/metadata.csv`
(inventário bibliográfico) e o relatório final.

## Como reproduzir

```
python scripts/01_extract_text.py      # extração PyMuPDF + limpeza estrutural
python scripts/02_preprocess.py        # tokens, stopwords, lemas spaCy, n-gramas
python scripts/03_select_k.py          # varrimento C_V k=3..20 (3 seeds)
python scripts/04_countries.py         # menções de países (pycountry + aliases)
python scripts/05_final_model.py 12 42 # modelo final k=12, seed 42
python scripts/06_figures.py           # 6 figuras + pyLDAvis
python scripts/07a_assemble_content.py # textos do relatório (lê data/topic_interpretations.json)
python scripts/07_report.py            # LDA_Report_JoaoLuis.docx
```

Nota: `07a_assemble_content.py` lê os parágrafos interpretativos de
`data/topic_interpretations.json`; o resultado consolidado está em
`results/report_content.json` — para regenerar o relatório basta correr
`07_report.py`, que lê apenas os CSV/JSON de `results/`.

## Outputs

Nota: os documentos Word gerados (`.docx`) **não são incluídos** no repositório;
são produzidos localmente ao correr os scripts. Os dados e figuras abaixo são os
que ficam versionados.

- `LDA_Report_JoaoLuis.docx` — relatório completo (33 pp., inglês), gerado por
  `07_report.py`: sumário executivo, metodologia, seleção de k, 12 tópicos com
  artigos representativos, evolução temporal, foco geográfico, robustez/limitações,
  Apêndice A (tópico dominante por artigo) e Apêndice B (matriz 125×12).
- `results/doc_topic_matrix.csv` — matriz documento×tópico completa.
- `results/dominant_topics.csv` — tópico dominante + proporção por artigo.
- `results/topic_terms.csv` — top-20 termos e pesos por tópico.
- `results/coherence_sweep.csv` — varrimento C_V (média±dp por k).
- `results/topic_year_share.csv` — quota média por tópico×ano.
- `results/country_mentions.csv` — nº de artigos que mencionam cada país.
- `results/figures/` — 6 figuras PNG (200 dpi) usadas no relatório.
- `results/pyldavis.html` — visualização interativa dos tópicos.
- `results/final_model/` — modelo gensim + dicionário (recarregáveis).

## Decisões de limpeza relevantes

- Remoção de listas de referências, rodapés/cabeçalhos repetidos e carimbos
  IEEE Xplore ("Authorized licensed use…").
- Remoção de citações in-text ("Autor et al.", parênteses com anos) para
  evitar apelidos de autores como termos de tópicos.
- Filtro de linhas em espanhol (3 revistas bilingues: Sidorov 2021, Soto 2022,
  Tuñoque-Morante 2024) + stopwords multilingues.
- Correção de lemas: datum→data, PCSs→pcs, KPIs→kpi.
