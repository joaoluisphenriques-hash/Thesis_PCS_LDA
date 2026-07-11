# -*- coding: utf-8 -*-
"""
Consolida os números necessários aos dois documentos (apêndice EN + contexto
PT) em results/doc_data.json — evita transcrição manual de valores.
"""
import json

import pandas as pd

from r00_common import (LAYER_ORDER, PERIOD_LABELS, RESULTS, THESIS_TABLE41)

LAYER_TITLES = {
    "operational": "Operational core (T8, T6, T7, T0)",
    "institutional": "Institutional & strategic (T10, T3, T9)",
    "technology": "Technology enablers (T1, T5, T2, T11)",
    "meta": "Meta-research (T4)",
}


def table_to_rows(df, group_field="group"):
    """CSV mean±sd -> lista de dicts ordenada por camada."""
    rows = []
    for group in df[group_field].unique():
        sub = df[df[group_field] == group].set_index("layer")
        for layer in LAYER_ORDER:
            rows.append({
                "group": group,
                "layer": layer,
                "layer_title": LAYER_TITLES[layer],
                "cells": [sub.loc[layer, p] for p in PERIOD_LABELS],
            })
    return rows


def main():
    summary = pd.read_csv(RESULTS / "runs_summary.csv")
    headline = json.loads((RESULTS / "headline.json").read_text(encoding="utf-8"))
    refcheck = json.loads((RESULTS / "reference_check.json").read_text(encoding="utf-8"))

    no_crossover = summary.loc[~summary.crossover_last_period, "run_id"].tolist()
    inst_declines = summary.loc[summary.inst_delta <= 0,
                                ["run_id", "inst_delta"]]
    n_terms_by_variant = (summary.groupby("variant")["n_terms"]
                          .first().to_dict())

    data = {
        "headline": headline,
        "reference_check": {
            "max_abs_dev_pp": refcheck["max_abs_dev_pp"],
            "ref_cv_same_method": refcheck["ref_cv_same_method"],
        },
        "thesis_table41": {
            layer: [float(THESIS_TABLE41.loc[layer, p]) for p in PERIOD_LABELS]
            for layer in LAYER_ORDER
        },
        "layer_titles": LAYER_TITLES,
        "period_labels": PERIOD_LABELS,
        "exceptions": {
            "no_crossover_runs": no_crossover,
            "inst_decline_runs": [
                {"run_id": r.run_id, "inst_delta": round(float(r.inst_delta), 1)}
                for r in inst_declines.itertuples()
            ],
        },
        "n_terms_by_variant": {k: int(v) for k, v in n_terms_by_variant.items()},
        "cv_range_all_runs": [round(float(summary.cv.min()), 3),
                              round(float(summary.cv.max()), 3)],
        "op_delta_all": [round(float(summary.op_delta.min()), 1),
                         round(float(summary.op_delta.max()), 1)],
        "table_seeds": table_to_rows(pd.read_csv(RESULTS / "table_seeds.csv")),
        "table_k": table_to_rows(pd.read_csv(RESULTS / "table_k.csv")),
        "table_prepro": table_to_rows(pd.read_csv(RESULTS / "table_prepro.csv")),
    }
    (RESULTS / "doc_data.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print("doc_data.json escrito.")
    print("Sem crossover:", no_crossover)
    print("Inst em queda:", inst_declines.run_id.tolist())


if __name__ == "__main__":
    main()
