# -*- coding: utf-8 -*-
"""
Passo 3 — Seleção do número de tópicos K por coerência C_V.

- Dicionário gensim com filter_extremes(no_below=5, no_above=0.5)
  (Ferdinand et al., 2024).
- Varrimento K = 3..14; para cada K, 3 inicializações (seeds) para medir
  média ± desvio-padrão da coerência C_V (Röder et al., 2015).
- Modelo: gensim LdaModel (Variational Bayes), alpha=0.1, eta=0.01,
  passes=20, iterations=400.
- A estimação de probabilidades para a C_V é feita numa única passagem
  sobre o corpus para todos os modelos (CoherenceModel.for_models).
- Outputs: results/coherence_sweep.csv + results/topics_by_k.txt.

NOTA Windows: todo o código está dentro de main() com o guard
if __name__ == '__main__' porque a CoherenceModel usa multiprocessing.
"""
import json
import time
from pathlib import Path

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel

BASE = Path(__file__).resolve().parents[1]   # raiz do repositório
RESULTS = BASE / "results"

K_RANGE = list(range(3, 21))
SEEDS = [42, 7, 13]
ALPHA, ETA = 0.1, 0.01
PASSES, ITERATIONS = 20, 400


def main():
    RESULTS.mkdir(exist_ok=True)
    tokens_map = json.loads(
        (BASE / "data" / "tokens.json").read_text(encoding="utf-8")
    )
    doc_ids = sorted(tokens_map)
    texts = [tokens_map[d] for d in doc_ids]

    dictionary = Dictionary(texts)
    before = len(dictionary)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    print(f"Dicionário: {before} -> {len(dictionary)} termos após filtragem",
          flush=True)
    corpus = [dictionary.doc2bow(t) for t in texts]

    # --- treinar todos os modelos -------------------------------------
    t0 = time.time()
    models, keys = [], []
    log = open(RESULTS / "topics_by_k.txt", "w", encoding="utf-8")
    for k in K_RANGE:
        for seed in SEEDS:
            lda = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=k,
                random_state=seed,
                alpha=ALPHA,
                eta=ETA,
                passes=PASSES,
                iterations=ITERATIONS,
                chunksize=64,
                eval_every=None,
            )
            models.append(lda)
            keys.append((k, seed))
            log.write(f"\n===== k={k} seed={seed} =====\n")
            for t in range(k):
                terms = ", ".join(w for w, _ in lda.show_topic(t, topn=12))
                log.write(f"  T{t}: {terms}\n")
            log.flush()
            print(f"treinado k={k:2d} seed={seed:2d} ({time.time()-t0:.0f}s)",
                  flush=True)
    log.close()

    # --- coerência C_V: uma passagem de estimação para todos ----------
    print("A estimar probabilidades C_V (passagem única)...", flush=True)
    try:
        cm = CoherenceModel.for_models(
            models, dictionary=dictionary, texts=texts, coherence="c_v"
        )
        comparisons = cm.compare_models(models)
        cvs = [pair[1] for pair in comparisons]
    except Exception as e:
        print(f"for_models falhou ({e}); fallback por modelo", flush=True)
        cvs = []
        for (k, seed), m in zip(keys, models):
            cv = CoherenceModel(
                model=m, texts=texts, dictionary=dictionary, coherence="c_v"
            ).get_coherence()
            cvs.append(cv)
            print(f"  C_V k={k} seed={seed}: {cv:.4f} ({time.time()-t0:.0f}s)",
                  flush=True)

    rows = [
        {"k": k, "seed": seed, "c_v": cv}
        for (k, seed), cv in zip(keys, cvs)
    ]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "coherence_sweep_raw.csv", index=False)
    summary = df.groupby("k")["c_v"].agg(["mean", "std"]).reset_index()
    summary.columns = ["k", "cv_mean", "cv_std"]
    summary.to_csv(RESULTS / "coherence_sweep.csv", index=False)
    print("\n=== RESUMO ===")
    print(summary.to_string(index=False))
    best = summary.loc[summary.cv_mean.idxmax()]
    print(f"\nMelhor K por C_V médio: {int(best.k)} (C_V={best.cv_mean:.4f})")
    print(f"Tempo total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
