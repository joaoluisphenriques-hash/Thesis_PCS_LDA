# -*- coding: utf-8 -*-
"""
Passo 6 — Figuras do relatório (formato do REPORT_TIPO).

Requer: results/ do passo 5 + results/topic_labels.json
        {"0": "rótulo do T0", "1": "...", ...}

Figuras (PNG, 200 dpi) em results/figures/:
  fig1_coherence.png      — C_V vs k (média ± dp, 3 seeds) + vlines
  fig2_wordclouds.png     — grelha de wordclouds (um painel por tópico)
  fig3_prevalence.png     — barras horizontais de prevalência
  fig4_heatmap.png        — heatmap tópico x ano (quota média %)
  fig5_papers_year.png    — nº de artigos por ano
  fig6_countries.png      — top-15 países mencionados
Extra: pyldavis.html (visualização interativa)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from wordcloud import WordCloud

BASE = Path(__file__).resolve().parents[1]   # raiz do repositório
RESULTS = BASE / "results"
FIG = RESULTS / "figures"
FIG.mkdir(exist_ok=True)

labels = {int(k): v for k, v in json.loads(
    (RESULTS / "topic_labels.json").read_text(encoding="utf-8")).items()}
stability = json.loads((RESULTS / "stability.json").read_text())
K = stability["k"]
CHOSEN_SEED = stability["best_seed"]

plt.rcParams.update({"font.size": 10, "figure.dpi": 200})


def fig1_coherence():
    df = pd.read_csv(RESULTS / "coherence_sweep.csv")
    kmax = int(df.loc[df.cv_mean.idxmax(), "k"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(df.k, df.cv_mean, yerr=df.cv_std, marker="o", capsize=4,
                color="#1f77b4", lw=1.5)
    ax.axvline(K, color="red", ls="--", lw=1.2, label=f"chosen k={K}")
    if kmax != K:
        ax.axvline(kmax, color="green", ls=":", lw=1.2, label=f"C_V max k={kmax}")
    ax.set_xlabel("Number of topics (k)")
    ax.set_ylabel("C_V coherence")
    ax.set_xticks(df.k)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_coherence.png", bbox_inches="tight")
    plt.close(fig)


def fig2_wordclouds(lda):
    import textwrap
    cols = 4 if K > 6 else 3
    rows = int(np.ceil(K / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.7 * rows))
    axes = np.atleast_1d(axes).ravel()
    for t in range(K):
        freqs = dict(lda.show_topic(t, topn=40))
        wc = WordCloud(width=640, height=480, background_color="white",
                       colormap="viridis", random_state=42)
        wc.generate_from_frequencies(freqs)
        axes[t].imshow(wc, interpolation="bilinear")
        title = "\n".join(textwrap.wrap(f"T{t}: {labels[t]}", width=38))
        axes[t].set_title(title, fontsize=10)
        axes[t].axis("off")
    for ax in axes[K:]:
        ax.axis("off")
    fig.tight_layout(h_pad=2.0)
    fig.savefig(FIG / "fig2_wordclouds.png", bbox_inches="tight")
    plt.close(fig)


def fig3_prevalence():
    ov = pd.read_csv(RESULTS / "topic_overview.csv").sort_values("prevalence")
    names = [f"T{t}: {labels[t]}" for t in ov.topic]
    colors = plt.cm.tab10(np.linspace(0, 1, K))[ov.topic.values]
    fig, ax = plt.subplots(figsize=(9, 0.55 * K + 1.5))
    ax.barh(names, ov.prevalence * 100, color=colors)
    ax.set_xlabel("Mean prevalence in corpus (%)")
    ax.set_title(f"Topic prevalence (k={K}, n=125)")
    for i, v in enumerate(ov.prevalence * 100):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_prevalence.png", bbox_inches="tight")
    plt.close(fig)


def fig4_heatmap():
    ys = pd.read_csv(RESULTS / "topic_year_share.csv")
    years = ys.year.astype(int)
    mat = ys[[f"T{t}" for t in range(K)]].values.T * 100
    fig, ax = plt.subplots(figsize=(11, 0.5 * K + 2))
    im = ax.imshow(mat, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"T{t}: {labels[t]}" for t in range(K)], fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean topic share (%)")
    ax.set_title("Topic evolution over time (heatmap of mean topic share per year)")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def fig5_papers_year():
    meta = pd.read_csv(BASE / "data" / "metadata.csv")
    counts = meta.year.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(counts.index.astype(int), counts.values, color="#1a3a6b")
    ax.set_xlabel("Publication year")
    ax.set_ylabel("# papers")
    ax.set_xticks(counts.index.astype(int))
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_papers_year.png", bbox_inches="tight")
    plt.close(fig)


def fig6_countries():
    cc = pd.read_csv(RESULTS / "country_mentions.csv").head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(cc.country, cc.n_articles, color="#2a9d8f")
    ax.set_xlabel("# papers mentioning")
    ax.set_title("Geographic focus of the corpus (top 15)")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_countries.png", bbox_inches="tight")
    plt.close(fig)


def pyldavis(lda, dictionary):
    try:
        import pyLDAvis
        import pyLDAvis.gensim_models as gm
        tokens_map = json.loads(
            (BASE / "data" / "tokens.json").read_text(encoding="utf-8"))
        texts = [tokens_map[d] for d in sorted(tokens_map)]
        corpus = [dictionary.doc2bow(t) for t in texts]
        vis = gm.prepare(lda, corpus, dictionary, mds="mmds", sort_topics=False)
        pyLDAvis.save_html(vis, str(RESULTS / "pyldavis.html"))
        print("pyldavis.html gerado")
    except Exception as e:
        print(f"pyLDAvis falhou (não crítico): {e}")


def main():
    lda = LdaModel.load(str(RESULTS / "final_model" / "lda_final.model"))
    dictionary = Dictionary.load(str(RESULTS / "final_model" / "dictionary.dict"))
    fig1_coherence()
    fig2_wordclouds(lda)
    fig3_prevalence()
    fig4_heatmap()
    fig5_papers_year()
    fig6_countries()
    print("6 figuras geradas em", FIG)
    pyldavis(lda, dictionary)


if __name__ == "__main__":
    main()
