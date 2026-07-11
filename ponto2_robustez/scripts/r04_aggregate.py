# -*- coding: utf-8 -*-
"""
Agregação da análise de robustez + figuras (estilo das figuras da tese).

Lê results/runs/*.json e produz em results/:
  reference_check.json       — validação: o modelo final da tese, avaliado com
                               a maquinaria deste apêndice, reproduz a Tabela 4.1
  runs_summary.csv           — uma linha por modelo variante
  layer_period_all_runs.csv  — formato longo (run × camada × período)
  table_seeds.csv            — média ± dp por camada×período (Eixo A)
  table_k.csv                — média ± dp por k (Eixo B; k=12 = 5 seeds da tese)
  table_prepro.csv           — média por variante de pré-processamento (Eixo C)
  headline.json              — indicadores globais para o apêndice
  figures/figR1..figR4       — trajetórias por eixo + deltas
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from r00_common import (BASE, FIG_DIR, LAYER_ORDER, PERIOD_LABELS,
                        REF_TOPIC_LAYER, RESULTS, RUNS_DIR, THESIS_TABLE41,
                        indicators, layer_period_table, load_corpus_tokens,
                        load_reference, load_years, theta_matrix)

plt.rcParams.update({"font.size": 10, "figure.dpi": 200})

LAYER_TITLES = {
    "operational": "Operational core",
    "institutional": "Institutional & strategic",
    "technology": "Technology enablers",
    "meta": "Meta-research",
}
X = np.arange(len(PERIOD_LABELS))


def reference_check():
    """Avalia o modelo final GUARDADO da tese com a maquinaria deste apêndice;
    deve reproduzir a Tabela 4.1 (valida a agregação camada×período)."""
    doc_ids, texts = load_corpus_tokens()
    years = load_years(doc_ids)
    ref_model, ref_dict = load_reference()
    corpus = [ref_dict.doc2bow(t) for t in texts]
    theta = theta_matrix(ref_model, corpus, 12)
    table, _ = layer_period_table(theta, years, REF_TOPIC_LAYER, 12)
    ind = indicators(table)
    # C_V do modelo de referência com a MESMA função usada nos 32 modelos
    # variantes (âncora de escala; o valor absoluto difere ligeiramente do
    # reportado no Cap. 3, que usou CoherenceModel.for_models/compare_models)
    from r00_common import coherence_cv
    ref_cv = coherence_cv(ref_model, texts, ref_dict)
    out = {
        "ref_cv_same_method": ref_cv,
        "table_pct": {l: {p: round(float(table.loc[l, p]), 1)
                          for p in PERIOD_LABELS} for l in LAYER_ORDER},
        "thesis_table41": {l: {p: float(THESIS_TABLE41.loc[l, p])
                               for p in PERIOD_LABELS} for l in LAYER_ORDER},
        "max_abs_dev_pp": ind["max_abs_dev_pp"],
    }
    (RESULTS / "reference_check.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"reference_check: desvio máximo à Tabela 4.1 = "
          f"{ind['max_abs_dev_pp']:.2f} pp")
    return table


def load_runs():
    recs = []
    for f in sorted(RUNS_DIR.glob("*.json")):
        recs.append(json.loads(f.read_text(encoding="utf-8")))
    return recs


def tidy(recs):
    flat, long_rows = [], []
    for r in recs:
        row = {k: v for k, v in r.items()
               if k not in ("table_pct", "layers_by_topic")}
        for layer in LAYER_ORDER:
            for p in PERIOD_LABELS:
                row[f"{layer}|{p}"] = r["table_pct"][layer][p]
                long_rows.append({
                    "run_id": r["run_id"], "axis": r["axis"],
                    "variant": r["variant"], "k": r["k"], "seed": r["seed"],
                    "layer": layer, "period": p,
                    "share_pct": r["table_pct"][layer][p],
                })
        flat.append(row)
    return pd.DataFrame(flat), pd.DataFrame(long_rows)


def mean_sd_table(long_df, mask, label):
    sub = long_df[mask]
    g = sub.groupby(["layer", "period"])["share_pct"].agg(["mean", "std"])
    rows = []
    for layer in LAYER_ORDER:
        row = {"group": label, "layer": layer}
        for p in PERIOD_LABELS:
            m, s = g.loc[(layer, p), "mean"], g.loc[(layer, p), "std"]
            row[p] = f"{m:.1f} ± {s:.1f}"
            row[f"{p}_mean"] = round(float(m), 2)
            row[f"{p}_sd"] = round(float(s), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_axis(long_df, runs_mask, fname, title, by=None):
    """2×2 painéis (um por camada); linhas finas por run ou média±dp por grupo;
    referência = Tabela 4.1 da tese (tracejado preto)."""
    sub = long_df[runs_mask]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, layer in zip(axes.ravel(), LAYER_ORDER):
        ldata = sub[sub.layer == layer]
        if by is None:
            for rid, g in ldata.groupby("run_id"):
                y = g.set_index("period").loc[PERIOD_LABELS, "share_pct"]
                ax.plot(X, y, color="#1f77b4", alpha=0.35, lw=1)
        else:
            palette = plt.cm.tab10(np.linspace(0, 1, 10))
            for ci, (gname, g) in enumerate(ldata.groupby(by)):
                piv = g.pivot_table(index="period", values="share_pct",
                                    aggfunc=["mean", "std"])
                m = piv[("mean", "share_pct")].loc[PERIOD_LABELS].values
                s = piv[("std", "share_pct")].loc[PERIOD_LABELS].values
                ax.plot(X, m, marker="o", ms=3.5, lw=1.5,
                        color=palette[ci], label=str(gname))
                ax.fill_between(X, m - s, m + s, color=palette[ci], alpha=0.15)
        ref = THESIS_TABLE41.loc[layer, PERIOD_LABELS].values
        ax.plot(X, ref, color="black", ls="--", lw=1.8, marker="s", ms=4,
                label="Thesis Table 4.1")
        ax.set_title(LAYER_TITLES[layer], fontsize=10)
        ax.set_xticks(X)
        ax.set_xticklabels(PERIOD_LABELS, fontsize=8)
        ax.set_ylabel("Topic mass (%)")
        ax.grid(alpha=0.3)
    handles, lab = axes.ravel()[0].get_legend_handles_labels()
    uniq = dict(zip(lab, handles))
    fig.legend(uniq.values(), uniq.keys(), loc="lower center",
               ncol=min(len(uniq), 5), frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)


def plot_deltas(summary, fname):
    """Variação (último − primeiro período) por camada, todos os modelos."""
    delta_cols = {"operational": "op_delta", "institutional": "inst_delta",
                  "technology": "tech_delta", "meta": "meta_delta"}
    axis_colors = {"A_seeds": "#1f77b4", "B_k": "#d62728", "C_prepro": "#2a9d8f"}
    axis_labels = {"A_seeds": "Seeds (k=12)", "B_k": "k ∈ {10, 14}",
                   "C_prepro": "Preprocessing"}
    rng = np.random.RandomState(0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for li, layer in enumerate(LAYER_ORDER):
        for axname, g in summary.groupby("axis"):
            yjit = li + rng.uniform(-0.18, 0.18, len(g))
            ax.scatter(g[delta_cols[layer]], yjit, s=28, alpha=0.75,
                       color=axis_colors[axname], label=axis_labels[axname],
                       edgecolors="white", lw=0.5)
        ref_delta = (THESIS_TABLE41.loc[layer, PERIOD_LABELS[-1]]
                     - THESIS_TABLE41.loc[layer, PERIOD_LABELS[0]])
        ax.scatter([ref_delta], [li], marker="D", s=70, color="black",
                   zorder=5, label="Thesis Table 4.1")
    ax.axvline(0, color="grey", lw=1)
    ax.set_yticks(range(len(LAYER_ORDER)))
    ax.set_yticklabels([LAYER_TITLES[l] for l in LAYER_ORDER])
    ax.set_xlabel("Change in layer share, 2023–2026 vs 2007–2014 (pp)")
    handles, lab = ax.get_legend_handles_labels()
    uniq = dict(zip(lab, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, axis="x")
    ax.set_title("First-to-last-period change in layer share across all "
                 "robustness runs")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)


def main():
    reference_check()
    recs = load_runs()
    summary, long_df = tidy(recs)
    summary.to_csv(RESULTS / "runs_summary.csv", index=False,
                   encoding="utf-8-sig")
    long_df.to_csv(RESULTS / "layer_period_all_runs.csv", index=False,
                   encoding="utf-8-sig")

    # tabelas agregadas -----------------------------------------------------
    t_seeds = mean_sd_table(long_df, long_df.axis == "A_seeds",
                            "10 seeds, k=12")
    t_seeds.to_csv(RESULTS / "table_seeds.csv", index=False,
                   encoding="utf-8-sig")

    thesis_seeds = [42, 7, 13, 21, 99]
    k_tables = []
    k_tables.append(mean_sd_table(
        long_df, (long_df.axis == "B_k") & (long_df.k == 10), "k=10"))
    k_tables.append(mean_sd_table(
        long_df, (long_df.axis == "A_seeds") & long_df.seed.isin(thesis_seeds),
        "k=12"))
    k_tables.append(mean_sd_table(
        long_df, (long_df.axis == "B_k") & (long_df.k == 14), "k=14"))
    pd.concat(k_tables).to_csv(RESULTS / "table_k.csv", index=False,
                               encoding="utf-8-sig")

    prepro_tables = []
    for v in ["unigrams", "vocab_loose", "vocab_strict", "no_domain_stop"]:
        prepro_tables.append(mean_sd_table(
            long_df, (long_df.axis == "C_prepro") & (long_df.variant == v), v))
    pd.concat(prepro_tables).to_csv(RESULTS / "table_prepro.csv", index=False,
                                    encoding="utf-8-sig")

    # indicadores globais ---------------------------------------------------
    stability_seeds = json.loads(
        (RESULTS / "stability_seeds.json").read_text(encoding="utf-8"))
    headline = {
        "n_runs": len(summary),
        "n_op_declines": int((summary.op_delta < 0).sum()),
        "n_op_monotone": int(summary.op_monotone_decline.sum()),
        "n_inst_rises": int((summary.inst_delta > 0).sum()),
        "n_crossover_last_period": int(summary.crossover_last_period.sum()),
        "n_tech_rises": int((summary.tech_delta > 0).sum()),
        "n_meta_rises": int((summary.meta_delta > 0).sum()),
        "corr_with_thesis_mean": float(summary.corr_with_thesis_table.mean()),
        "corr_with_thesis_min": float(summary.corr_with_thesis_table.min()),
        "mean_top10_overlap_with_ref": float(
            summary.mean_top10_overlap_with_ref.mean()),
        "by_axis": {
            ax: {
                "n": int(len(g)),
                "op_declines": int((g.op_delta < 0).sum()),
                "inst_rises": int((g.inst_delta > 0).sum()),
                "crossover": int(g.crossover_last_period.sum()),
                "op_monotone": int(g.op_monotone_decline.sum()),
                "op_delta_range": [float(g.op_delta.min()),
                                   float(g.op_delta.max())],
                "inst_delta_range": [float(g.inst_delta.min()),
                                     float(g.inst_delta.max())],
            } for ax, g in summary.groupby("axis")
        },
        "seed_stability": stability_seeds,
    }
    (RESULTS / "headline.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8")

    # figuras ----------------------------------------------------------------
    plot_axis(long_df, long_df.axis == "A_seeds", "figR1_seeds.png",
              "Layer trajectories across 10 random seeds (k = 12)")
    kmask = (long_df.axis == "B_k") | (
        (long_df.axis == "A_seeds") & long_df.seed.isin(thesis_seeds))
    plot_axis(long_df, kmask, "figR2_k.png",
              "Layer trajectories for k = 10, 12, 14 (mean ± sd across 5 seeds)",
              by="k")
    plot_axis(long_df, long_df.axis == "C_prepro", "figR3_prepro.png",
              "Layer trajectories under alternative preprocessing "
              "(mean ± sd across 3 seeds, k = 12)", by="variant")
    plot_deltas(summary, "figR4_deltas.png")

    print(json.dumps(headline, indent=2)[:1500])
    print("\nAgregação concluída.")


if __name__ == "__main__":
    main()
