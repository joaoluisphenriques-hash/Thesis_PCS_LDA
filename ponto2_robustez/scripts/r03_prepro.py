# -*- coding: utf-8 -*-
"""
Eixo C — Robustez a escolhas de pré-processamento (k=12, 3 seeds).

Variantes:
  unigrams      — sem colocações: os n-gramas do pipeline de referência são
                  desfeitos nos unigramas constituintes (testa a escolha de
                  Phrases/bigramas+trigramas).
  vocab_loose   — dicionário permissivo: no_below=2, no_above=0.9
                  (testa o filtro de extremos 5/0.5).
  vocab_strict  — dicionário estrito: no_below=10, no_above=0.4.
  no_domain_stop— pré-processamento re-executado de raiz a partir de
                  data/raw_text SEM a lista de stopwords de domínio
                  académico (testa a decisão de limpeza mais discricionária
                  do pipeline). Mantém limpeza estrutural, filtro de idioma,
                  lematização spaCy, correção de acrónimos e n-gramas.
"""
import json
import re
from pathlib import Path

from r00_common import (BASE, ROB, build_dictionary, evaluate_model,
                        load_corpus_tokens, load_reference, load_years,
                        train_lda)

SEEDS = [42, 7, 13]
K = 12
NODOMAIN_TOKENS = ROB / "data_variants" / "tokens_nodomainstop.json"


# --------------------------------------------------------------------------
# Variante no_domain_stop: réplica de scripts/02_preprocess.py sem DOMAIN_STOP
# --------------------------------------------------------------------------
def build_nodomainstop_tokens():
    import spacy
    from gensim.models.phrases import Phrases, Phraser
    from nltk.corpus import stopwords

    STOP = set(stopwords.words("english"))
    for lang in ("spanish", "portuguese", "french", "italian", "german"):
        STOP |= set(stopwords.words(lang))
    # fragmentos de má des-hifenização (artefactos de extração, mantidos)
    STOP |= {"tion", "tions", "ment", "ments", "sion", "sions", "ing", "ity",
             "ted", "ters", "ble", "ical", "ally", "cally", "ness", "ance",
             "ence", "pro", "con", "tem", "sys", "of_the", "in_the"}
    # substantivos espanhóis que escapam ao filtro de linhas (idioma, mantidos)
    STOP |= {"puerto", "puertos", "buque", "buques", "navio", "navios",
             "mercancía", "mercancias", "investigación", "análisis",
             "gestión", "artículo", "resultados", "estudio", "tecnología"}
    # NOTA: a lista DOMAIN_STOP (boilerplate académico + verbos genéricos)
    # do pipeline de referência é deliberadamente OMITIDA nesta variante.

    ACRONYM_FIX = {"pcss": "pcs", "kpis": "kpi", "msws": "msw", "nsws": "nsw",
                   "pcses": "pcs", "icts": "ict"}
    TOKEN_RE = re.compile(r"^[a-z]+$")
    URL_RE = re.compile(r"https?://\S+|www\.\S+|\S+@\S+")
    HYPHEN_BREAK_RE = re.compile(r"(\w)-\s+(\w)")
    CITATION_ETAL_RE = re.compile(
        r"\b[A-ZÀ-Ž][a-zà-ž]+(?:\s+(?:and|&|e)\s+[A-ZÀ-Ž][a-zà-ž]+)?\s+et\s+al\.?")
    CITATION_PAREN_RE = re.compile(r"\([^()]*\b(?:19|20)\d{2}[a-z]?\b[^()]*\)")
    ES_FUNC = {"las", "los", "del", "por", "para", "una", "como", "este",
               "esta", "entre", "sobre", "con", "más", "que", "ser", "son",
               "está", "también", "donde", "cuando", "hasta", "desde",
               "hacia", "puerto", "puertos", "buque", "buques", "carga",
               "sistema", "sistemas", "investigación", "resultados",
               "análisis", "artículo"}
    WORD_RE = re.compile(r"[a-záéíóúñüà-ž]+")

    def drop_non_english_paragraphs(text):
        kept = []
        for parag in text.split("\n"):
            words = WORD_RE.findall(parag.lower())
            if len(words) >= 6:
                ratio = sum(1 for w in words if w in ES_FUNC) / len(words)
                if ratio > 0.10:
                    continue
            kept.append(parag)
        return "\n".join(kept)

    def clean_text(text):
        text = drop_non_english_paragraphs(text)
        text = HYPHEN_BREAK_RE.sub(r"\1\2", text)
        text = CITATION_ETAL_RE.sub(" ", text)
        text = CITATION_PAREN_RE.sub(" ", text)
        text = URL_RE.sub(" ", text)
        text = re.sub(r"\s+", " ", text)
        return text.lower()

    files = sorted((BASE / "data" / "raw_text").glob("doc*.txt"))
    print(f"no_domain_stop: a lematizar {len(files)} documentos…", flush=True)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp.max_length = 300_000

    doc_ids, texts = [], []
    for f in files:
        doc_ids.append(f.stem)
        texts.append(clean_text(f.read_text(encoding="utf-8")))

    def norm(lemma):
        lemma = "data" if lemma == "datum" else lemma
        return ACRONYM_FIX.get(lemma, lemma)

    tokenized = []
    for i, doc in enumerate(nlp.pipe(texts, batch_size=8), 1):
        toks = [norm(t.lemma_.lower()) for t in doc
                if TOKEN_RE.match(t.text) and len(t.text) >= 3
                and t.lemma_.lower() not in STOP and t.text not in STOP
                and len(t.lemma_) >= 3]
        tokenized.append(toks)
        if i % 25 == 0:
            print(f"  {i}/{len(texts)}", flush=True)

    bigram = Phraser(Phrases(tokenized, min_count=5, threshold=10))
    tokens_bi = [bigram[t] for t in tokenized]
    trigram = Phraser(Phrases(tokens_bi, min_count=5, threshold=10))
    tokens_tri = [trigram[t] for t in tokens_bi]

    NODOMAIN_TOKENS.write_text(
        json.dumps(dict(zip(doc_ids, tokens_tri))), encoding="utf-8")
    print("no_domain_stop: tokens gravados.", flush=True)


def main():
    doc_ids, texts_ref = load_corpus_tokens()
    years = load_years(doc_ids)
    ref_model, ref_dict = load_reference()

    # tokens da variante unigrams: desfazer n-gramas nos constituintes
    texts_uni = [[part for tok in doc for part in tok.split("_")]
                 for doc in texts_ref]

    if not NODOMAIN_TOKENS.exists():
        build_nodomainstop_tokens()
    ids_nd, texts_nd = load_corpus_tokens(NODOMAIN_TOKENS)
    assert ids_nd == doc_ids

    variants = [
        ("unigrams", texts_uni, dict(no_below=5, no_above=0.5)),
        ("vocab_loose", texts_ref, dict(no_below=2, no_above=0.9)),
        ("vocab_strict", texts_ref, dict(no_below=10, no_above=0.4)),
        ("no_domain_stop", texts_nd, dict(no_below=5, no_above=0.5)),
    ]

    for name, texts, filt in variants:
        dictionary = build_dictionary(texts, **filt)
        corpus = [dictionary.doc2bow(t) for t in texts]
        print(f"\n[{name}] dicionário: {len(dictionary)} termos", flush=True)
        for seed in SEEDS:
            run_id = f"C_{name}_s{seed}"
            model = train_lda(corpus, dictionary, K, seed)
            rec = evaluate_model(run_id, "C_prepro", name, K, seed,
                                 model, dictionary, corpus, texts, years,
                                 doc_ids, ref_model, ref_dict)
            print(f"{run_id}: cv={rec['cv']:.4f} "
                  f"op_delta={rec['op_delta']:+.1f}pp "
                  f"inst_delta={rec['inst_delta']:+.1f}pp "
                  f"crossover={rec['crossover_last_period']}", flush=True)


if __name__ == "__main__":
    main()
