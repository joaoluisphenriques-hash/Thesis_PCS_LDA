# -*- coding: utf-8 -*-
"""
Passo 2 — Pré-processamento do corpus para LDA.

Pipeline (alinhado com Ferdinand et al. 2024 e prática padrão gensim):
  1. Filtro de idioma: remoção de parágrafos não-ingleses (3 artigos de
     revistas bilingues incluem a versão espanhola no mesmo PDF).
  2. Limpeza: des-hifenização (hífen + qualquer whitespace), remoção de
     citações in-text ("Autor et al.", parênteses com anos), URLs/emails.
  3. Normalização: minúsculas, tokenização (apenas tokens alfabéticos, len>=3).
  4. Stopwords: inglês + espanhol/português/francês/italiano/alemão (nltk)
     + lista de domínio académico + fragmentos de hifenização.
  5. Lematização: spaCy en_core_web_sm (parser e NER desativados);
     correção de plurais de acrónimos (pcss->pcs, kpis->kpi).
  6. Colocações: bigramas e trigramas estatisticamente frequentes
     (gensim Phrases, min_count=5, threshold=10).

Output: data/tokens.json (lista de listas de tokens por doc_id) +
        data/token_stats.txt (frequências para auditoria de stopwords).
"""
import json
import re
from collections import Counter
from pathlib import Path

import spacy
from gensim.models.phrases import Phrases, Phraser
from nltk.corpus import stopwords

BASE = Path(__file__).resolve().parents[1]   # raiz do repositório
TXT_DIR = BASE / "data" / "raw_text"
OUT_TOKENS = BASE / "data" / "tokens.json"
OUT_STATS = BASE / "data" / "token_stats.txt"

# --- stopwords -------------------------------------------------------------
STOP = set(stopwords.words("english"))
for lang in ("spanish", "portuguese", "french", "italian", "german"):
    STOP |= set(stopwords.words(lang))

# Fragmentos típicos de má des-hifenização em PDFs
STOP |= {"tion", "tions", "ment", "ments", "sion", "sions", "ing", "ity",
         "ted", "ters", "ble", "ical", "ally", "cally", "ness", "ance",
         "ence", "pro", "con", "tem", "sys", "of_the", "in_the"}

# Substantivos espanhóis do domínio que escapam ao filtro de linhas
STOP |= {"puerto", "puertos", "buque", "buques", "navio", "navios",
         "mercancía", "mercancias", "investigación", "análisis",
         "gestión", "artículo", "resultados", "estudio", "tecnología"}

# Plurais de acrónimos que a lematização não resolve
ACRONYM_FIX = {"pcss": "pcs", "kpis": "kpi", "msws": "msw", "nsws": "nsw",
               "pcses": "pcs", "icts": "ict"}

# Termos académicos/estruturais de baixa informação, artefactos de extração
# de PDF e termos editoriais — iterado após inspeção das frequências.
DOMAIN_STOP = {
    # academic boilerplate
    "study", "studies", "paper", "research", "article", "author", "authors",
    "journal", "review", "literature", "analysis", "result", "results",
    "finding", "findings", "conclusion", "conclusions", "introduction",
    "abstract", "keywords", "section", "table", "figure", "fig", "chapter",
    "approach", "method", "methods", "methodology", "case", "cases",
    "example", "examples", "discussion", "hypothesis", "sample",
    "respondent", "respondents", "questionnaire", "interview", "interviews",
    "survey", "item", "items", "issue", "issues", "vol", "volume", "page",
    "pages", "doi", "issn", "isbn", "http", "https", "www", "com", "org",
    "elsevier", "springer", "ieee", "mdpi", "emerald", "taylor", "francis",
    "wiley", "copyright", "license", "licensee", "creative", "commons",
    "publisher", "publishing", "press", "proceedings", "conference",
    "university", "department", "faculty", "institute", "email", "mail",
    "received", "revised", "accepted", "available", "online", "access",
    "cite", "citation", "cited", "reference", "references", "appendix",
    # verbos/adverbios genéricos frequentes em texto académico
    "also", "however", "therefore", "thus", "moreover", "furthermore",
    "although", "whereas", "hence", "may", "might", "could", "would",
    "should", "shall", "must", "well", "one", "two", "three", "first",
    "second", "third", "new", "based", "use", "used", "using", "usage",
    "user", "users", "provide", "provided", "provides", "providing",
    "include", "included", "includes", "including", "within", "among",
    "along", "since", "due", "many", "much", "several", "various",
    "different", "important", "significant", "main", "specific", "general",
    "high", "higher", "low", "lower", "large", "small", "number", "level",
    "levels", "way", "ways", "need", "needs", "needed", "make", "makes",
    "made", "making", "take", "taken", "give", "given", "show", "shown",
    "shows", "present", "presented", "presents", "describe", "described",
    "consider", "considered", "identify", "identified", "propose",
    "proposed", "develop", "developed", "achieve", "achieved", "obtain",
    "obtained", "related", "regarding", "according", "respectively",
    "et", "al", "etc", "eg", "ie",
}
STOP |= DOMAIN_STOP

TOKEN_RE = re.compile(r"^[a-z]+$")
URL_RE = re.compile(r"https?://\S+|www\.\S+|\S+@\S+")
HYPHEN_BREAK_RE = re.compile(r"(\w)-\s+(\w)")
# citações in-text: "Surname et al." / "Surname and Surname (2020)" /
# qualquer parêntese que contenha um ano
CITATION_ETAL_RE = re.compile(
    r"\b[A-ZÀ-Ž][a-zà-ž]+(?:\s+(?:and|&|e)\s+[A-ZÀ-Ž][a-zà-ž]+)?\s+et\s+al\.?"
)
CITATION_PAREN_RE = re.compile(r"\([^()]*\b(?:19|20)\d{2}[a-z]?\b[^()]*\)")

# Palavras espanholas (funcionais + substantivos do domínio marítimo) para
# deteção de linhas não-inglesas em PDFs bilingues
ES_FUNC = {"las", "los", "del", "por", "para", "una", "como", "este", "esta",
           "entre", "sobre", "con", "más", "que", "ser", "son", "está",
           "también", "donde", "cuando", "hasta", "desde", "hacia",
           "puerto", "puertos", "buque", "buques", "carga", "sistema",
           "sistemas", "investigación", "resultados", "análisis", "artículo"}
WORD_RE = re.compile(r"[a-záéíóúñüà-ž]+")


def drop_non_english_paragraphs(text: str) -> str:
    """Remove linhas com densidade alta de palavras espanholas (revistas
    bilingues que duplicam o texto em espanhol no mesmo PDF)."""
    kept = []
    for parag in text.split("\n"):
        words = WORD_RE.findall(parag.lower())
        if len(words) >= 6:
            ratio = sum(1 for w in words if w in ES_FUNC) / len(words)
            if ratio > 0.10:
                continue
        kept.append(parag)
    return "\n".join(kept)


def clean_text(text: str) -> str:
    text = drop_non_english_paragraphs(text)
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)  # digitali-\nzation -> digitalization
    text = CITATION_ETAL_RE.sub(" ", text)
    text = CITATION_PAREN_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def main():
    files = sorted(TXT_DIR.glob("doc*.txt"))
    print(f"{len(files)} documentos")

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp.max_length = 300_000

    doc_ids, texts = [], []
    for f in files:
        doc_ids.append(f.stem)
        texts.append(clean_text(f.read_text(encoding="utf-8")))

    def norm(lemma: str) -> str:
        lemma = "data" if lemma == "datum" else lemma
        return ACRONYM_FIX.get(lemma, lemma)

    tokenized = []
    for i, doc in enumerate(nlp.pipe(texts, batch_size=8), 1):
        toks = [
            norm(t.lemma_.lower())
            for t in doc
            if TOKEN_RE.match(t.text)
            and len(t.text) >= 3
            and t.lemma_.lower() not in STOP
            and t.text not in STOP
            and len(t.lemma_) >= 3
        ]
        tokenized.append(toks)
        if i % 25 == 0:
            print(f"  lematizados {i}/{len(texts)}")

    # Colocações: bigramas e depois trigramas
    bigram = Phraser(Phrases(tokenized, min_count=5, threshold=10))
    tokens_bi = [bigram[t] for t in tokenized]
    trigram = Phraser(Phrases(tokens_bi, min_count=5, threshold=10))
    tokens_tri = [trigram[t] for t in tokens_bi]

    OUT_TOKENS.write_text(
        json.dumps(dict(zip(doc_ids, tokens_tri))), encoding="utf-8"
    )

    # Estatísticas para auditoria
    freq = Counter(tok for doc in tokens_tri for tok in doc)
    ngrams = [(w, c) for w, c in freq.most_common() if "_" in w]
    with open(OUT_STATS, "w", encoding="utf-8") as f:
        f.write("=== TOP 100 TERMOS ===\n")
        for w, c in freq.most_common(100):
            f.write(f"{w}\t{c}\n")
        f.write("\n=== TOP 60 N-GRAMAS ===\n")
        for w, c in ngrams[:60]:
            f.write(f"{w}\t{c}\n")

    total = sum(freq.values())
    print(f"Tokens totais: {total} | vocabulário: {len(freq)}")
    print(f"N-gramas detetados: {len(ngrams)}")
    print("Top 25:", [w for w, _ in freq.most_common(25)])


if __name__ == "__main__":
    main()
