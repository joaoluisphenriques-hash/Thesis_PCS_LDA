# -*- coding: utf-8 -*-
"""
Passo 5 — Modelo LDA final (K escolhido) + outputs tabulares.

Uso: python 05_final_model.py <K> [seed_final]

- Treina o modelo com K tópicos para 5 seeds; por omissão seleciona o de
  maior C_V; se seed_final for dado, fixa esse seed (seleção por
  interpretabilidade, documentada no relatório).
- Estabilidade entre seeds: ARI médio entre as atribuições de tópico
  dominante e sobreposição média dos top-10 termos (emparelhamento
  Hungarian sobre distância de Jensen-Shannon entre distribuições
  tópico-palavra).
- Outputs em results/:
    final_model/ (modelo gensim + dicionário)
    doc_topic_matrix.csv  (125 x K, minimum_probability=0)
    dominant_topics.csv   (tópico dominante + proporção por artigo)
    topic_terms.csv       (top-20 termos e pesos por tópico)
    topic_overview.csv    (nº artigos dominantes + prevalência média)
    topic_year_share.csv  (quota média por tópico x ano)
    stability.json        (C_V por seed, ARI médio, overlap médio)
"""
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import adjusted_rand_score

BASE = Path(__file__).resolve().parents[1]   # raiz do repositório
RESULTS = BASE / "results"
SEEDS = [42, 7, 13, 21, 99]
ALPHA, ETA = 0.1, 0.01
PASSES, ITERATIONS = 20, 400


def top_term_overlap(m1, m2, k, topn=10):
    """Sobreposição média dos top-N termos após emparelhamento Hungarian."""
    phi1 = m1.get_topics()  # k x V
    phi2 = m2.get_topics()
    cost = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            cost[i, j] = jensenshannon(phi1[i], phi2[j])
    ri, ci = linear_sum_assignment(cost)
    overlaps = []
    for i, j in zip(ri, ci):
        t1 = {w for w, _ in m1.show_topic(i, topn=topn)}
        t2 = {w for w, _ in m2.show_topic(j, topn=topn)}
        overlaps.append(len(t1 & t2) / topn)
    return float(np.mean(overlaps))


def main():
    k = int(sys.argv[1])
    tokens_map = json.loads((BASE / "data" / "tokens.json").read_text(encoding="utf-8"))
    doc_ids = sorted(tokens_map)
    texts = [tokens_map[d] for d in doc_ids]
    meta = pd.read_csv(BASE / "data" / "metadata.csv").set_index("doc_id")

    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(t) for t in texts]
    n_terms = len(dictionary)
    print(f"K={k} | dicionário={n_terms} termos", flush=True)

    # --- treinar 5 seeds ----------------------------------------------
    t0 = time.time()
    models = {}
    for seed in SEEDS:
        models[seed] = LdaModel(
            corpus=corpus, id2word=dictionary, num_topics=k,
            random_state=seed, alpha=ALPHA, eta=ETA,
            passes=PASSES, iterations=ITERATIONS, chunksize=64,
            eval_every=None,
        )
        print(f"  seed {seed} treinado ({time.time()-t0:.0f}s)", flush=True)

    cm = CoherenceModel.for_models(
        list(models.values()), dictionary=dictionary, texts=texts,
        coherence="c_v",
    )
    cvs = dict(zip(SEEDS, (p[1] for p in cm.compare_models(list(models.values())))))
    if len(sys.argv) > 2:
        best_seed = int(sys.argv[2])
        selection = "interpretabilidade (painel de juízes)"
    else:
        best_seed = max(cvs, key=cvs.get)
        selection = "C_V máximo"
    lda = models[best_seed]
    print(f"C_V por seed: { {s: round(c,4) for s,c in cvs.items()} }")
    print(f"Seed escolhida: {best_seed} (C_V={cvs[best_seed]:.4f}, por {selection})",
          flush=True)

    # --- estabilidade --------------------------------------------------
    dom = {}
    for seed, m in models.items():
        theta = np.zeros((len(corpus), k))
        for i, bow in enumerate(corpus):
            for t, p in m.get_document_topics(bow, minimum_probability=0.0):
                theta[i, t] = p
        dom[seed] = theta.argmax(axis=1)
    aris = [adjusted_rand_score(dom[a], dom[b]) for a, b in combinations(SEEDS, 2)]
    overlaps = [top_term_overlap(models[a], models[b], k) for a, b in combinations(SEEDS, 2)]
    stability = {
        "cv_by_seed": {str(s): cvs[s] for s in SEEDS},
        "best_seed": best_seed,
        "seed_selection": selection,
        "mean_ari": float(np.mean(aris)),
        "mean_top10_overlap": float(np.mean(overlaps)),
        "n_terms": n_terms,
        "k": k,
        "alpha": ALPHA, "eta": ETA,
        "passes": PASSES, "iterations": ITERATIONS,
    }
    (RESULTS / "stability.json").write_text(json.dumps(stability, indent=2))
    print(f"ARI médio: {np.mean(aris):.3f} | overlap top-10 médio: {np.mean(overlaps):.3f}")

    # --- matriz documento-tópico ---------------------------------------
    theta = np.zeros((len(corpus), k))
    for i, bow in enumerate(corpus):
        for t, p in lda.get_document_topics(bow, minimum_probability=0.0):
            theta[i, t] = p
    cols = [f"T{t}" for t in range(k)]
    dt = pd.DataFrame(theta, index=doc_ids, columns=cols)
    dt.index.name = "doc_id"
    dt.join(meta[["authors", "year", "title"]]).to_csv(
        RESULTS / "doc_topic_matrix.csv", encoding="utf-8-sig"
    )

    # --- tópico dominante ----------------------------------------------
    dom_df = pd.DataFrame({
        "doc_id": doc_ids,
        "dominant_topic": theta.argmax(axis=1),
        "proportion": theta.max(axis=1),
    }).set_index("doc_id").join(meta[["authors", "year", "title"]])
    dom_df.to_csv(RESULTS / "dominant_topics.csv", encoding="utf-8-sig")

    # --- termos de topo -------------------------------------------------
    rows = []
    for t in range(k):
        for rank, (w, p) in enumerate(lda.show_topic(t, topn=20), 1):
            rows.append({"topic": t, "rank": rank, "term": w, "weight": p})
    pd.DataFrame(rows).to_csv(RESULTS / "topic_terms.csv", index=False,
                              encoding="utf-8-sig")

    # --- visão geral -----------------------------------------------------
    overview = pd.DataFrame({
        "topic": range(k),
        "n_dominant": [int((theta.argmax(axis=1) == t).sum()) for t in range(k)],
        "prevalence": theta.mean(axis=0),
    })
    overview.to_csv(RESULTS / "topic_overview.csv", index=False)

    # --- quota por ano ---------------------------------------------------
    years = meta.loc[doc_ids, "year"].values
    ys = []
    for y in sorted(set(years)):
        mask = years == y
        ys.append([y, int(mask.sum())] + list(theta[mask].mean(axis=0)))
    pd.DataFrame(ys, columns=["year", "n_docs"] + cols).to_csv(
        RESULTS / "topic_year_share.csv", index=False
    )

    # --- guardar modelo ---------------------------------------------------
    outdir = RESULTS / "final_model"
    outdir.mkdir(exist_ok=True)
    lda.save(str(outdir / "lda_final.model"))
    dictionary.save(str(outdir / "dictionary.dict"))

    print("\n=== TÓPICOS FINAIS ===")
    for t in range(k):
        terms = ", ".join(w for w, _ in lda.show_topic(t, topn=12))
        n = int((theta.argmax(axis=1) == t).sum())
        print(f"T{t} ({n} docs, {theta.mean(axis=0)[t]*100:.1f}%): {terms}")


if __name__ == "__main__":
    main()
