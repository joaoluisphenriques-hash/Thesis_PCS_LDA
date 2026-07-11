# -*- coding: utf-8 -*-
"""Monta results/report_content.json: textos autorais + interpretações
por tópico (data/topic_interpretations.json)."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]   # raiz do repositório
RES = BASE / "results"
INTERPS_IN = BASE / "data" / "topic_interpretations.json"

raw = json.loads(INTERPS_IN.read_text(encoding="utf-8"))
interps = raw["result"] if "result" in raw else raw
labels = json.loads((RES / "topic_labels.json").read_text(encoding="utf-8"))

# correção de nome vindo do nome de ficheiro
interps["0"] = interps["0"].replace(
    "Ali Mubder Mubder e Fredriksson", "Mubder and Fredriksson"
)

topics = {t: {"label": labels[t], "interpretation": interps[t]}
          for t in labels}

content = {
    "prepared_for": "João Luís",
    "date_line": "July 2026",
    "subtitle": "Latent Dirichlet Allocation over the port-digitalization literature",
    "description": "Thematic mapping of 125 full-text peer-reviewed articles on Port Community Systems, maritime single windows and smart ports.",
    "k_word": "twelve",

    "exec_intro": "This report presents a Latent Dirichlet Allocation (LDA) topic model of 125 full-text academic articles (2007–2026) on the digitalization of seaports — Port Community Systems (PCS), maritime single windows, smart ports and related themes. Following a C_V coherence sweep over k = 3–20 and a structured interpretability assessment, a twelve-topic solution was selected. The twelve topics are:",

    "key_findings": [
        {"lead": "Foundation of the field.", "text": "The documentary backbone of the literature is formed by Port Community Systems for import/export flows (T8) together with the two single-window streams (T6, T7): jointly they account for roughly a third of the corpus and dominate the 2009–2016 period, when EU reporting-formalities directives and national single-window programmes drove the research agenda."},
        {"lead": "Emerging themes.", "text": "Governance and sustainable digital transformation (T10) is the largest single cluster (22 dominant articles, 16.5% prevalence) and rises steadily from 2017 onwards, while cybersecurity (T1) and bibliometric research-trend mapping (T4) concentrate almost entirely in 2024–2026 — the newest fronts of the field."},
        {"lead": "Technology waves.", "text": "Blockchain for trade documentation (T5) forms a small but sharply delineated cluster concentrated around 2020–2023, consistent with a technology-enthusiasm wave that has since receded; platform business models (T11) remain a two-article niche connecting the field to the platform-economy literature."},
    ],

    "introduction": "Latent Dirichlet Allocation (Blei, Ng and Jordan, 2003) is a generative probabilistic model that represents each document as a mixture of latent topics and each topic as a probability distribution over terms. Applied to a scientific corpus, it provides a reproducible, corpus-wide map of the thematic structure of a literature, complementing manual review with an unsupervised, quantitative account of which research streams exist, how large they are and how they evolve over time (Ferdinand et al., 2024). The objective of this analysis is to identify the latent thematic structure of a 125-article corpus on seaport digitalization — Port Community Systems, maritime single windows, smart ports and adjacent streams — in order to structure the literature review of the associated dissertation, assign each article to its dominant research stream, and characterize the temporal and geographic contours of the field.",

    "corpus_paragraph": "The corpus comprises 125 peer-reviewed journal articles and conference papers published between 2007 and 2026, collected for the dissertation's literature review. Full text was extracted from the PDF files with PyMuPDF. Extraction was followed by structural cleaning: reference lists were removed (cut at the last 'References'/'Bibliography' heading in the second half of each document); per-page furniture (repeated headers, footers and download/copyright stamps) was dropped; hyphenation across line breaks was repaired; in-text citations (parenthetical author–year strings and 'Author et al.' mentions) and URLs were removed; and non-English passages in three bilingual journals were filtered out. Table 1 summarizes the corpus and model configuration.",

    "preprocessing_bullets": [
        {"lead": "Normalization.", "text": "Lower-casing and tokenization retaining alphabetic tokens with at least three characters."},
        {"lead": "Stopwords.", "text": "Standard English stopwords (NLTK) extended with an iteratively built academic/domain list (e.g., paper, study, journal, publisher names) and with Spanish, Portuguese, French, Italian and German stopword lists as a safeguard for residual non-English text."},
        {"lead": "Lemmatization.", "text": "spaCy (en_core_web_sm) lemmatization, with post-correction of acronym plurals (e.g., PCSs → pcs) and of the 'data' lemma."},
        {"lead": "Collocations.", "text": "Statistically salient bigrams and trigrams detected with the gensim Phrases model (min_count = 5, threshold = 10), so that multiword expressions such as port_community, single_window, smart_port, bill_lading and digital_transformation are treated as single terms."},
    ],

    "dictionary_paragraph": "The preprocessed tokens were mapped to a gensim dictionary, filtered by removing terms occurring in fewer than 5 documents or in more than 50% of documents (no_below = 5, no_above = 0.5), following Ferdinand et al. (2024). The filter yields 5,002 terms and, importantly, removes corpus-wide vocabulary (port, system, data, information), which forces the model to discriminate documents on more specific terminology. Each document was then encoded as a bag-of-words vector (doc2bow).",

    "inference_paragraph": "Topics were inferred with gensim's LdaModel, which implements online Variational Bayes. The final model uses k = 12 topics, a symmetric document–topic prior alpha = 0.1, a topic–word prior beta (eta) = 0.01, 20 passes over the corpus with 400 inference iterations per document batch (chunksize = 64). Five random initializations (seeds 42, 7, 13, 21, 99) were trained; the reported model fixes seed 42, selected through a structured interpretability comparison of the candidate solutions, whose C_V (0.4833) lies within one standard deviation of the best initialization (0.4971). Fixing the seed makes the reported model exactly reproducible from the accompanying scripts.",

    "k_selection_paragraph": "The number of topics k is the main analyst-controlled parameter of LDA. It was selected with a coherence sweep: for each k from 3 to 20, three models with different random seeds were trained and their C_V coherence (Röder, Both and Hinneburg, 2015) computed against the corpus; C_V correlates best with human judgements of topic quality among the standard coherence measures. Following the heuristic of Ferdinand et al. (2024) — choose the highest coherence before the curve flattens or drops — the sweep was complemented by a qualitative interpretability assessment of the candidate solutions, since coherence alone is a guiding metric rather than a decision rule.",

    "selection_justification": "Figure 1 shows the sweep. Mean C_V rises steeply up to k ≈ 10, reaches a plateau at k = 10–12 and then oscillates within noise up to k = 20. The global maximum sits at k = 14 (C_V = 0.5028), but with the highest seed variance of the sweep (± 0.0286), while k = 12 attains a statistically indistinguishable coherence (0.4920) with the lowest variance (± 0.0075) — the most reproducible solution. A structured qualitative comparison of the k = 10, 12 and 14 solutions across seeds confirmed the choice of k = 12: at k = 10 distinct streams remain merged (governance with platform business models; terminal operations never separates), while at k = 14 the extra capacity fragments into single-case shards and unlabelable residual topics. The twelve-topic solution offers the cleanest separation of the field's recognized research streams — single-window, PCS operations, blockchain, cybersecurity, governance/sustainability, smart-port evaluation — while every topic remains labelable.",

    "results_explainer": "Each topic is a probability distribution over the 5,002 dictionary terms, and each article is a mixture of topics. Topic labels were assigned by joint inspection of the highest-probability terms and of the articles most strongly associated with each topic. Figure 2 shows the dominant terms of the twelve topics; Table 3 summarizes their size (number of articles for which the topic is dominant) and prevalence (mean share of the topic across all 125 documents). The subsections below present the topics in descending order of prevalence.",

    "topics": topics,

    "temporal_interpretation": "The temporal pattern in Figure 4 traces the field's maturation. The documentary-infrastructure streams dominate the first decade: PCS import/export flows (T8) peak between 2009 and 2016, and the two single-window topics (T6, T7) surge from 2011 to 2016, in the wake of Directive 2010/65/EU on reporting formalities and the associated national implementation programmes. From 2017 onwards the centre of gravity shifts to the strategic-institutional layer: governance and sustainable digital transformation (T10) grows steadily and holds the largest share of the 2024–2026 period, while cybersecurity (T1) and bibliometric mapping (T4) emerge as the newest streams, concentrated in the last three years. Blockchain (T5) exhibits a clear wave pattern centred on 2020–2023. Port-call coordination (T0), whose lone 2007 article makes the earliest column, re-emerges in the 2020s around Just-in-Time port calls — an old coordination problem revisited with new digital instruments.",

    "temporal_note": "The 2007–2011 columns contain a single article each and 2026 is a partial year of in-press items (see Figure 5), so single documents fully determine those cells; shares before 2012 should be read as illustrative only.",

    "geographic_paragraph": "China (27 articles), the United States (26) and Singapore (23) are the most frequently mentioned countries, reflecting their role as reference cases of port scale and digital maturity. The next tier is distinctly European — Italy, the United Kingdom, Croatia, the Netherlands, Germany and Spain — mirroring the corpus's strong orientation toward EU-funded PCS and single-window case studies; the prominence of Croatia (17) and of the Adriatic cluster more broadly matches the single-window implementation streams (T6, T7), much of whose literature documents Croatian and Montenegrin national programmes. Colombia (11) and Indonesia (10) anchor the Latin-American and Southeast-Asian case-study strands. Note that counts measure mentions in article bodies (at least two mentions per article), a proxy for geographic focus rather than for author affiliation.",

    "robustness_bullets": [
        {"lead": "Reproducibility.", "text": "All results derive from deterministic scripts with fixed random seeds (final model: seed 42, alpha = 0.1, eta = 0.01, k = 12); the full pipeline — extraction, preprocessing, sweep, model, figures, report — can be re-run end-to-end from the accompanying code."},
        {"lead": "Model selection.", "text": "k was chosen where the coherence plateau begins (k = 12: C_V = 0.4920 ± 0.0075, the lowest seed variance of the sweep) and validated by a structured interpretability comparison against k = 10 and k = 14; the final initialization was selected among five seeds for interpretability, with coherence within one standard deviation of the best seed."},
    ],

    "limitations_bullets": [
        {"lead": "Stability.", "text": "Topic solutions vary across random initializations: the mean adjusted Rand index between the dominant-topic partitions of the five seeds is 0.135, and the mean top-10 term overlap of matched topics is 34%. The reported topics are one defensible reading of the corpus, not a unique decomposition; document-level assignments (Appendix A) should be read as probabilistic tendencies."},
        {"lead": "Full text.", "text": "Modeling full texts (rather than abstracts) captures richer signal but also methods vocabulary and case-specific place names; despite aggressive cleaning, topics T2 and T7 retain visible case-study and survey-methods flavor, and two topics (T4, T11) rest on four or fewer dominant articles."},
        {"lead": "Coherence.", "text": "C_V coherence is a guiding heuristic, not an optimality criterion: the k = 10–20 plateau means several k values are defensible, and the final choice of k = 12 rests partly on qualitative interpretability judgements."},
        {"lead": "Corpus coverage.", "text": "Three bilingual articles had their non-English text filtered; the earliest years (2007–2011) contribute one article each; and 2026 is incomplete — temporal readings at the edges of the window are correspondingly fragile."},
    ],

    "conclusions": "The LDA decomposition organizes the 125-article corpus into three readable layers. A documentary-regulatory layer — PCS import/export flows (T8), single-window architecture and implementation (T6, T7) and blockchain documentation (T5) — constitutes the historical core of the field and reflects two decades of effort to dematerialize trade and reporting formalities. A technical-operational layer — port-call coordination (T0), operational optimization and service quality (T2), smart-port evaluation (T9) and cybersecurity (T1) — treats digitalization as an instrument of, and a risk to, port performance. A strategic-institutional layer — adoption barriers and stakeholder dynamics (T3), governance and sustainable digital transformation (T10), platform business models (T11) and research-trend mapping (T4) — frames digitalization as an organizational and policy transformation. Temporally, the corpus documents a clear trajectory from the documentary layer (dominant 2009–2016) toward the strategic-institutional one (dominant since 2017), with cybersecurity and meta-scientific reflection as the newest fronts. For the dissertation, the twelve topics provide both a defensible structure for the literature-review chapters and an empirical map of where the field's open questions — governance capacity, sustainability alignment, cyber-resilience — are currently concentrated.",

    "appendixA_explainer": "The table below lists, for each of the 125 articles, its dominant topic (the topic with the highest share in the article's topic mixture) and the corresponding proportion. Articles are sorted by topic and, within each topic, by descending proportion. Topic labels: " + "; ".join(f"T{t} — {labels[str(t)]}" for t in range(12)) + ".",

    "appendixB_explainer": "The matrix below gives the full document–topic distribution of the final model (minimum_probability = 0): each row is an article, each column a topic, each cell the topic's share of that article; rows sum to 1 and the dominant cell is set in bold. Studies are sorted alphabetically.",
}

out = RES / "report_content.json"
out.write_text(json.dumps(content, indent=1, ensure_ascii=False), encoding="utf-8")
print("report_content.json escrito:", out)
print("Tópicos com interpretação:", sorted(topics.keys(), key=int))
