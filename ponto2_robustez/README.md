# Ponto 2 da Scientific Review — Apêndice de Robustez do Topic Modelling

Resposta ao ponto 2 do orientador (*"Topic-modelling instability should have
stronger implications for the claims"*): um apêndice de robustez que verifica
se o **padrão de quatro camadas** da Tabela 4.1 da tese — declínio do núcleo
operacional, subida e ultrapassagem pela camada institucional-estratégica —
se mantém para lá da configuração única escolhida (k=12, seed 42).

## Desenho da análise

Três eixos de perturbação, exatamente os pedidos na revisão:

| Eixo | Variação | Modelos |
|------|----------|---------|
| A — Seeds | 10 inicializações aleatórias, k=12, pipeline de referência | 10 |
| B — Nº de tópicos | k = 10 e k = 14, 5 seeds cada (k=12 vem do Eixo A) | 10 |
| C — Pré-processamento | 4 variantes × 3 seeds, k=12 | 12 |

Variantes de pré-processamento (Eixo C):
- `unigrams` — sem colocações (n-gramas desfeitos nos constituintes);
- `vocab_loose` — filtro de dicionário permissivo (`no_below=2, no_above=0.9`);
- `vocab_strict` — filtro estrito (`no_below=10, no_above=0.4`);
- `no_domain_stop` — pipeline re-executado de raiz a partir de `data/raw_text`
  **sem** a lista de stopwords de domínio académico (a decisão de limpeza mais
  discricionária do pipeline de referência).

## Método de mapeamento tópico→camada

Cada tópico de um modelo variante herda a camada do tópico do modelo final da
tese que lhe está mais próximo — distância de Jensen-Shannon entre
distribuições tópico-palavra, restringida ao vocabulário comum e renormalizada.
O mapeamento usa **apenas** as distribuições de palavras (semântica); a tabela
camada×período é depois recalculada a partir do θ do próprio modelo variante,
pelo que o sinal temporal nunca é herdado do modelo de referência.

Validação da maquinaria: o modelo final guardado da tese, avaliado por este
código, reproduz a Tabela 4.1 com desvio máximo de 0.04 pp
(`results/reference_check.json`).

## Como reproduzir

```
cd ponto2_robustez/scripts
python r01_seeds.py     # Eixo A + estabilidade tópico vs camada
python r02_k.py         # Eixo B
python r03_prepro.py    # Eixo C (regenera tokens sem stopwords de domínio)
python r04_aggregate.py # agregação, tabelas, figuras, headline.json
```

## Outputs

- `results/runs/*.json` + `*_mapping.csv` — um registo por modelo variante
  (tabela camada×período, indicadores, mapeamento tópico→camada com JS e
  overlap de termos, para auditoria).
- `results/assignments/*.csv` — tópico e camada dominante por documento.
- `results/runs_summary.csv` — uma linha por modelo (32 modelos).
- `results/table_seeds.csv`, `table_k.csv`, `table_prepro.csv` — tabelas
  média ± dp prontas para o apêndice.
- `results/stability_seeds.json` — ARI ao nível do tópico vs ao nível da
  camada (10 seeds).
- `results/headline.json` — indicadores globais (nº de modelos com declínio
  operacional, crossover, etc.).
- `results/figures/figR1..figR4` — trajetórias por eixo + deltas.

O texto do apêndice em formato Word (`.docx`) não é incluído no repositório; os
resultados versionados são os dados e figuras de `results/` listados acima.
