# -*- coding: utf-8 -*-
"""
Helpers comuns da análise de robustez (Ponto 2 da Scientific Review).

Lógica central:
  1. Cada modelo variante (seed / k / pré-processamento) é treinado de raiz.
  2. Cada tópico do modelo variante é atribuído a uma das quatro camadas da
     tese (Tabela 4.1) herdando a camada do tópico de REFERÊNCIA mais próximo
     — distância de Jensen-Shannon entre distribuições tópico-palavra,
     restringida ao vocabulário comum e renormalizada.
  3. A tabela camada×período é recalculada a partir do θ do PRÓPRIO modelo
     variante. O mapeamento usa apenas semântica (distribuições de palavras);
     o sinal temporal nunca é herdado do modelo de referência.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
from scipy.spatial.distance import jensenshannon

BASE = Path(__file__).resolve().parents[2]   # 11_LDA_FINAL
ROB = Path(__file__).resolve().parents[1]    # ponto2_robustez
RESULTS = ROB / "results"
RUNS_DIR = RESULTS / "runs"
ASSIGN_DIR = RESULTS / "assignments"
FIG_DIR = RESULTS / "figures"

ALPHA, ETA = 0.1, 0.01
PASSES, ITERATIONS = 20, 400

# Camadas da Tabela 4.1 da tese (índices dos tópicos do modelo final k=12)
LAYERS = {
    "operational": [8, 6, 7, 0],
    "institutional": [10, 3, 9],
    "technology": [1, 5, 2, 11],
    "meta": [4],
}
REF_TOPIC_LAYER = {t: layer for layer, ts in LAYERS.items() for t in ts}
LAYER_ORDER = ["operational", "institutional", "technology", "meta"]

PERIODS = [(2007, 2014), (2015, 2018), (2019, 2022), (2023, 2026)]
PERIOD_LABELS = ["2007-2014", "2015-2018", "2019-2022", "2023-2026"]

# Tabela 4.1 da tese (percentagens) para comparação direta
THESIS_TABLE41 = pd.DataFrame(
    [[58.9, 57.9, 49.0, 32.2],
     [25.3, 30.6, 30.3, 35.7],
     [15.5, 8.8, 16.8, 24.2],
     [0.3, 2.7, 3.9, 7.9]],
    index=LAYER_ORDER, columns=PERIOD_LABELS,
)


def load_corpus_tokens(tokens_path=None):
    """doc_ids ordenados + listas de tokens (pré-processamento de referência)."""
    path = Path(tokens_path) if tokens_path else BASE / "data" / "tokens.json"
    tokens_map = json.loads(path.read_text(encoding="utf-8"))
    doc_ids = sorted(tokens_map)
    return doc_ids, [tokens_map[d] for d in doc_ids]


def load_years(doc_ids):
    meta = pd.read_csv(BASE / "data" / "metadata.csv").set_index("doc_id")
    return meta.loc[doc_ids, "year"].astype(int).values


def load_reference():
    """Modelo final da tese (k=12, seed 42) + dicionário."""
    ref_model = LdaModel.load(str(BASE / "results" / "final_model" / "lda_final.model"))
    ref_dict = Dictionary.load(str(BASE / "results" / "final_model" / "dictionary.dict"))
    return ref_model, ref_dict


def build_dictionary(texts, no_below=5, no_above=0.5):
    d = Dictionary(texts)
    d.filter_extremes(no_below=no_below, no_above=no_above)
    return d


def train_lda(corpus, dictionary, k, seed):
    return LdaModel(
        corpus=corpus, id2word=dictionary, num_topics=k,
        random_state=seed, alpha=ALPHA, eta=ETA,
        passes=PASSES, iterations=ITERATIONS, chunksize=64,
        eval_every=None,
    )


def theta_matrix(model, corpus, k):
    theta = np.zeros((len(corpus), k))
    for i, bow in enumerate(corpus):
        for t, p in model.get_document_topics(bow, minimum_probability=0.0):
            theta[i, t] = p
    return theta


def coherence_cv(model, texts, dictionary):
    cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary,
                        coherence="c_v", processes=1)
    return float(cm.get_coherence())


def _term_prob_matrix(model, dictionary, terms):
    """Matriz k×|terms| com as probabilidades de cada termo em cada tópico."""
    phi = model.get_topics()  # k x V
    token2id = dictionary.token2id
    idx = [token2id[t] for t in terms]
    return phi[:, idx]


def map_topics_to_layers(model, dictionary, ref_model, ref_dict, k):
    """Atribui cada tópico do modelo variante à camada do tópico de referência
    mais próximo (JS sobre vocabulário comum, renormalizado).

    Devolve (topic_layer: dict, mapping: DataFrame com diagnóstico)."""
    common = sorted(set(dictionary.token2id) & set(ref_dict.token2id))
    var_phi = _term_prob_matrix(model, dictionary, common)
    ref_phi = _term_prob_matrix(ref_model, ref_dict, common)
    # renormalizar sobre o vocabulário comum
    var_phi = var_phi / var_phi.sum(axis=1, keepdims=True)
    ref_phi = ref_phi / ref_phi.sum(axis=1, keepdims=True)

    ref_k = ref_phi.shape[0]
    topic_layer, rows = {}, []
    for i in range(k):
        dists = [jensenshannon(var_phi[i], ref_phi[j]) for j in range(ref_k)]
        j = int(np.argmin(dists))
        layer = REF_TOPIC_LAYER[j]
        topic_layer[i] = layer
        var_top = {w for w, _ in model.show_topic(i, topn=10)}
        ref_top = {w for w, _ in ref_model.show_topic(j, topn=10)}
        rows.append({
            "variant_topic": i,
            "matched_ref_topic": j,
            "layer": layer,
            "js_distance": float(dists[j]),
            "top10_overlap_with_ref": len(var_top & ref_top) / 10,
            "variant_top10": ", ".join(w for w, _ in model.show_topic(i, topn=10)),
            "ref_top10": ", ".join(w for w, _ in ref_model.show_topic(j, topn=10)),
        })
    return topic_layer, pd.DataFrame(rows), len(common)


def layer_period_table(theta, years, topic_layer, k):
    """Tabela camada×período: média (por documento) da massa de cada camada,
    em percentagem — a mesma agregação da Tabela 4.1 da tese."""
    layer_mass = {layer: np.zeros(theta.shape[0]) for layer in LAYER_ORDER}
    for t in range(k):
        layer_mass[topic_layer[t]] += theta[:, t]
    table = np.zeros((len(LAYER_ORDER), len(PERIODS)))
    for pi, (y0, y1) in enumerate(PERIODS):
        mask = (years >= y0) & (years <= y1)
        for li, layer in enumerate(LAYER_ORDER):
            table[li, pi] = layer_mass[layer][mask].mean() * 100
    df = pd.DataFrame(table, index=LAYER_ORDER, columns=PERIOD_LABELS)
    dominant_layer = np.array([
        LAYER_ORDER[int(np.argmax([layer_mass[l][i] for l in LAYER_ORDER]))]
        for i in range(theta.shape[0])
    ])
    return df, dominant_layer


def indicators(table):
    """Indicadores do padrão central da tese sobre uma tabela camada×período."""
    op = table.loc["operational"].values
    inst = table.loc["institutional"].values
    tech = table.loc["technology"].values
    meta = table.loc["meta"].values
    flat_v = table.values.flatten()
    flat_ref = THESIS_TABLE41.values.flatten()
    return {
        "op_first": float(op[0]), "op_last": float(op[-1]),
        "op_delta": float(op[-1] - op[0]),
        "op_monotone_decline": bool(np.all(np.diff(op) < 0)),
        "inst_first": float(inst[0]), "inst_last": float(inst[-1]),
        "inst_delta": float(inst[-1] - inst[0]),
        "crossover_last_period": bool(inst[-1] > op[-1]),
        "tech_delta": float(tech[-1] - tech[0]),
        "meta_delta": float(meta[-1] - meta[0]),
        "corr_with_thesis_table": float(np.corrcoef(flat_v, flat_ref)[0, 1]),
        "max_abs_dev_pp": float(np.max(np.abs(flat_v - flat_ref))),
    }


def evaluate_model(run_id, axis, variant, k, seed, model, dictionary, corpus,
                   texts, years, doc_ids, ref_model, ref_dict,
                   compute_cv=True):
    """Avalia um modelo variante e persiste o resultado em runs/ e assignments/."""
    theta = theta_matrix(model, corpus, k)
    topic_layer, mapping, n_common = map_topics_to_layers(
        model, dictionary, ref_model, ref_dict, k)
    table, dominant_layer = layer_period_table(theta, years, topic_layer, k)
    ind = indicators(table)
    cv = coherence_cv(model, texts, dictionary) if compute_cv else None

    record = {
        "run_id": run_id, "axis": axis, "variant": variant,
        "k": k, "seed": seed,
        "n_terms": len(dictionary), "n_common_vocab": n_common,
        "cv": cv,
        "mean_js_to_matched_ref": float(mapping["js_distance"].mean()),
        "mean_top10_overlap_with_ref": float(mapping["top10_overlap_with_ref"].mean()),
        "layers_by_topic": {str(t): l for t, l in topic_layer.items()},
        "table_pct": {layer: {p: float(table.loc[layer, p])
                              for p in PERIOD_LABELS}
                      for layer in LAYER_ORDER},
        **ind,
    }
    (RUNS_DIR / f"{run_id}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    mapping.insert(0, "run_id", run_id)
    mapping.to_csv(RUNS_DIR / f"{run_id}_mapping.csv", index=False,
                   encoding="utf-8-sig")
    pd.DataFrame({
        "doc_id": doc_ids,
        "dominant_topic": theta.argmax(axis=1),
        "dominant_layer": dominant_layer,
    }).to_csv(ASSIGN_DIR / f"{run_id}.csv", index=False)
    return record
