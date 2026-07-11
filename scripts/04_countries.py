# -*- coding: utf-8 -*-
"""
Passo 4 — Deteção de foco geográfico por menções de países no corpo do texto.

- Léxico: pycountry (nomes oficiais + comuns) + lista de aliases curada
  (UK, USA, Holanda, Coreia do Sul, etc.).
- Regex com fronteiras de palavra; contagem = nº de artigos que mencionam
  o país (>= MIN_MENTIONS menções para reduzir falsos positivos).
- Output: results/country_mentions.csv
"""
import csv
import re
from pathlib import Path

import pycountry

BASE = Path(__file__).resolve().parents[1]   # raiz do repositório
TXT_DIR = BASE / "data" / "raw_text"
OUT = BASE / "results" / "country_mentions.csv"
OUT.parent.mkdir(exist_ok=True)

MIN_MENTIONS = 2  # menções mínimas num artigo para contar

ALIASES = {
    "United States": ["USA", "U.S.", "US ", "United States of America",
                      "the United States"],
    "United Kingdom": ["UK", "U.K.", "Great Britain", "England", "Scotland"],
    "Netherlands": ["the Netherlands", "Holland"],
    "South Korea": ["Korea, Republic of", "Republic of Korea", "Korea"],
    "Russia": ["Russian Federation"],
    "Taiwan": ["Taiwan, Province of China", "Chinese Taipei"],
    "Vietnam": ["Viet Nam"],
    "Iran": ["Iran, Islamic Republic of"],
    "Tanzania": ["Tanzania, United Republic of"],
    "Turkey": ["Türkiye", "Turkiye"],
    "Czechia": ["Czech Republic"],
    "UAE": ["United Arab Emirates", "Dubai", "Abu Dhabi"],
    "Hong Kong": [],
    "Singapore": [],
}

# Nomes de países que colidem com termos comuns/entidades do domínio
EXCLUDE_NAMES = {"Jersey", "Georgia", "Jordan", "Chad", "Guinea", "Niger",
                 "Bar", "Man", "Isle of Man"}


def build_lexicon():
    lex = {}
    for c in pycountry.countries:
        display = getattr(c, "common_name", None) or c.name
        if display in EXCLUDE_NAMES:
            continue
        names = {c.name}
        if hasattr(c, "common_name"):
            names.add(c.common_name)
        # remover formas "X, Republic of" — cobertas por aliases
        names = {n for n in names if "," not in n}
        if names:
            lex.setdefault(display, set()).update(names)
    for display, extra in ALIASES.items():
        lex.setdefault(display, set()).add(display)
        lex[display].update(extra)
    # FIX 2026-07-05 (auditoria E-02): absorver entradas do pycountry cujo
    # display é alias de uma entrada curada (ex.: "Türkiye" -> Turkey,
    # "Russian Federation" -> Russia, "United Arab Emirates" -> UAE).
    # Sem isto, o mesmo país gera duas linhas no CSV (84 rótulos = 81 países).
    alias_to_display = {}
    for display, extra in ALIASES.items():
        for n in list(extra) + [display]:
            alias_to_display[n] = display
    for dup in list(lex.keys()):
        if dup in alias_to_display and alias_to_display[dup] != dup:
            target = alias_to_display[dup]
            lex.setdefault(target, set()).update(lex.pop(dup))
    # South Korea: remover match genérico "Korea" que apanha North Korea?
    # mantemos "Korea" — no corpus refere-se quase sempre à Coreia do Sul.
    return lex


def main():
    lex = build_lexicon()
    patterns = {
        display: re.compile(
            r"\b(?:" + "|".join(re.escape(n.strip()) for n in sorted(names, key=len, reverse=True)) + r")\b"
        )
        for display, names in lex.items()
    }

    files = sorted(TXT_DIR.glob("doc*.txt"))
    counts = {}       # país -> nº de artigos
    doc_hits = {}     # doc -> lista de países
    for f in files:
        text = f.read_text(encoding="utf-8")
        hits = []
        for display, pat in patterns.items():
            n = len(pat.findall(text))
            if n >= MIN_MENTIONS:
                counts[display] = counts.get(display, 0) + 1
                hits.append(display)
        doc_hits[f.stem] = hits

    rows = sorted(counts.items(), key=lambda x: -x[1])
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["country", "n_articles"])
        w.writerows(rows)

    print(f"{len(rows)} países detetados (>= {MIN_MENTIONS} menções/artigo)")
    print("Top 20:")
    for c, n in rows[:20]:
        print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
