# -*- coding: utf-8 -*-
"""
Eixo B — Robustez ao número de tópicos: k = 10 e k = 14 (5 seeds cada).

k = 12 é coberto pelo Eixo A (as 5 primeiras seeds coincidem com as da tese).
Cada tópico dos modelos k=10/k=14 herda a camada do tópico de referência
mais próximo (JS), e a tabela camada×período é recalculada.
"""
from r00_common import (evaluate_model, load_corpus_tokens, load_reference,
                        load_years, train_lda)

SEEDS = [42, 7, 13, 21, 99]
KS = [10, 14]


def main():
    doc_ids, texts = load_corpus_tokens()
    years = load_years(doc_ids)
    ref_model, ref_dict = load_reference()
    dictionary = ref_dict
    corpus = [dictionary.doc2bow(t) for t in texts]

    for k in KS:
        for seed in SEEDS:
            run_id = f"B_k{k}_s{seed}"
            model = train_lda(corpus, dictionary, k, seed)
            rec = evaluate_model(run_id, "B_k", f"k{k}", k, seed,
                                 model, dictionary, corpus, texts, years,
                                 doc_ids, ref_model, ref_dict)
            print(f"{run_id}: cv={rec['cv']:.4f} "
                  f"op_delta={rec['op_delta']:+.1f}pp "
                  f"inst_delta={rec['inst_delta']:+.1f}pp "
                  f"crossover={rec['crossover_last_period']}", flush=True)


if __name__ == "__main__":
    main()
