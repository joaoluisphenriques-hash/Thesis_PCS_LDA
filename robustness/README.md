# Robustness analysis of the four-layer pattern

This folder contains the robustness analysis reported in **Appendix D** of the thesis. It tests whether the layer-level finding of the review — the contraction of the operational core of the PCS literature and the rise of the institutional and strategic layer (Table 4.1 of the thesis) — survives the three perturbations to which an LDA pipeline is most exposed: the random seed, the number of topics *k*, and the preprocessing choices.

## Design

Thirty-two variant models were estimated with the configuration of the main pipeline (gensim `LdaModel`, online variational Bayes, α = 0.1, η = 0.01, 20 passes, 400 iterations), varying one ingredient at a time; the set includes a re-run of the retained configuration (k = 12, seed 42) as an internal control.

| Axis | Variation | Models |
|------|-----------|--------|
| A — Seeds | 10 random initialisations, k = 12, reference pipeline (5,002-term dictionary) | 10 |
| B — Number of topics | k = 10 and k = 14 (the two candidate solutions from the coherence sweep), 5 seeds each | 10 |
| C — Preprocessing | 4 alternative pipelines × 3 seeds, k = 12 | 12 |

Preprocessing variants (axis C): unigrams only (no collocation detection; 3,596 terms); permissive dictionary filter (`no_below=2, no_above=0.9`; 10,148 terms); strict filter (`no_below=10, no_above=0.4`; 2,465 terms); and a pipeline re-run from the raw text without the corpus-specific stop-word list (5,517 terms).

## Topic-to-layer mapping

Each topic of a variant model inherits the layer of its closest reference topic — the topic of the retained k = 12 model minimising the Jensen–Shannon distance between topic–word distributions, computed over the shared vocabulary and renormalised. The assignment uses only the word distributions; the layer-by-period table is then recomputed from the variant model's own document–topic matrix, so the temporal signal is never inherited from the reference solution. As a validation of the machinery, the retained model evaluated through this procedure reproduces Table 4.1 of the thesis within 0.05 percentage points (`results/reference_check.json`).

## Headline results

The operational core loses topic mass between 2007–2014 and 2023–2026 in all 32 variant models (−39.5 to −6.7 pp); the institutional and strategic layer gains share in 27 of 32 and overtakes the operational core in the final period in 28 of 32; layer-by-period tables correlate with Table 4.1 at r = 0.89 on average (minimum 0.48). Document-level assignments remain unstable at the topic level (mean ARI 0.127 across ten seeds) and only moderately more stable at the layer level (ARI 0.184; 60% dominant-layer agreement between seed pairs): the instability concerns the identity of individual topics, while the aggregate four-layer pattern is robust. See `results/headline.json` and `results/runs_summary.csv`.

## How to reproduce

From `robustness/scripts`, in order:

```
python r01_seeds.py       # axis A + topic- vs layer-level stability
python r02_k.py           # axis B
python r03_prepro.py      # axis C
python r04_aggregate.py   # aggregation, tables, figures, headline.json
python r07_fig_compact.py # composite summary figure (thesis Figure D.1)
```

Requirements: `gensim`, `spacy` (`en_core_web_sm`), `nltk`, `scikit-learn`, `scipy`, `pandas`, `matplotlib`. All seeds are fixed; results are exactly reproducible.

**Note on data:** the scripts read the tokenised corpus and dictionary produced by the main pipeline (`scripts/01…` in the repository root), and the `no_domain_stop` variant additionally re-tokenises the raw extracted text. The raw and tokenised full texts of the 125 papers are **not distributed** in this repository for copyright reasons; they can be regenerated from the original PDFs with the main pipeline's extraction step.

## Outputs

- `results/runs/*.json` + `*_mapping.csv` — one record per variant model: layer-by-period table, indicators, and the auditable topic-to-layer mapping (Jensen–Shannon distance, top-10 term overlap, top terms of both sides).
- `results/assignments/*.csv` — dominant topic and layer per document, per model.
- `results/runs_summary.csv` — one row per model (32 models).
- `results/table_seeds.csv`, `table_k.csv`, `table_prepro.csv` — mean ± sd layer-by-period tables per axis (the source data summarised in Appendix D and Figure D.1).
- `results/stability_seeds.json` — topic-level vs layer-level ARI across the ten seeds.
- `results/headline.json` — global indicators (declines, crossovers, correlations).
- `results/figures/` — per-axis trajectory figures and the composite summary figure (thesis Figure D.1).
