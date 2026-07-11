# -*- coding: utf-8 -*-
"""
Eixo A — Robustez a inicializações (seeds), k=12.

Treina 10 seeds com o pré-processamento e dicionário de referência (5.002
termos) e verifica se o padrão de quatro camadas da Tabela 4.1 se mantém.
Também calcula a estabilidade entre seeds ao nível do TÓPICO (ARI, como na
tese) e ao nível da CAMADA (ARI das atribuições de camada dominante), que é
a granularidade a que as afirmações centrais da tese são feitas.
"""
import json
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from r00_common import (ASSIGN_DIR, RESULTS, evaluate_model, load_corpus_tokens,
                        load_reference, load_years, train_lda)

SEEDS = [42, 7, 13, 21, 99, 1, 2, 3, 123, 2024]
K = 12


def main():
    doc_ids, texts = load_corpus_tokens()
    years = load_years(doc_ids)
    ref_model, ref_dict = load_reference()
    dictionary = ref_dict  # mesmo vocabulário da tese
    corpus = [dictionary.doc2bow(t) for t in texts]

    for seed in SEEDS:
        run_id = f"A_k12_s{seed}"
        model = train_lda(corpus, dictionary, K, seed)
        rec = evaluate_model(run_id, "A_seeds", "reference_prepro", K, seed,
                             model, dictionary, corpus, texts, years, doc_ids,
                             ref_model, ref_dict)
        print(f"{run_id}: cv={rec['cv']:.4f} op_delta={rec['op_delta']:+.1f}pp "
              f"inst_delta={rec['inst_delta']:+.1f}pp "
              f"crossover={rec['crossover_last_period']}", flush=True)

    # --- estabilidade entre seeds: tópico vs camada -----------------------
    dom_topic, dom_layer = {}, {}
    for seed in SEEDS:
        df = pd.read_csv(ASSIGN_DIR / f"A_k12_s{seed}.csv")
        dom_topic[seed] = df["dominant_topic"].values
        dom_layer[seed] = df["dominant_layer"].values
    ari_topic = [adjusted_rand_score(dom_topic[a], dom_topic[b])
                 for a, b in combinations(SEEDS, 2)]
    ari_layer = [adjusted_rand_score(dom_layer[a], dom_layer[b])
                 for a, b in combinations(SEEDS, 2)]
    # concordância simples: % de documentos com a mesma camada dominante
    agree_layer = [float(np.mean(dom_layer[a] == dom_layer[b]))
                   for a, b in combinations(SEEDS, 2)]
    out = {
        "seeds": SEEDS,
        "n_pairs": len(ari_topic),
        "mean_ari_topic": float(np.mean(ari_topic)),
        "mean_ari_layer": float(np.mean(ari_layer)),
        "mean_layer_agreement": float(np.mean(agree_layer)),
        "note": ("ARI ao nível do tópico replica a métrica da tese (0.135 "
                 "em 5 seeds); ARI/concordância ao nível da camada mede a "
                 "estabilidade à granularidade a que as conclusões são feitas."),
    }
    (RESULTS / "stability_seeds.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nARI topico={out['mean_ari_topic']:.3f} | "
          f"ARI camada={out['mean_ari_layer']:.3f} | "
          f"concordancia camada={out['mean_layer_agreement']:.3f}")


if __name__ == "__main__":
    main()
