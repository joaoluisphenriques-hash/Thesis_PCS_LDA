# -*- coding: utf-8 -*-
"""
Passo 1 — Extração de texto dos PDFs do corpus (LDA_PAPERS).

- Extrai o texto integral de cada PDF com PyMuPDF.
- Remove a secção de referências bibliográficas (corte na última ocorrência
  de um cabeçalho "References"/"Bibliography" na metade final do documento).
- Extrai metadados a partir do nome do ficheiro (autores, ano, título).
- Guarda um .txt por documento e um metadata.csv com o inventário.
"""
import csv
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

BASE = Path(__file__).resolve().parents[1]   # raiz do repositório
CORPUS_DIR = BASE / "LDA_PAPERS"              # coloque aqui os PDFs (não versionado)
OUT_DIR = BASE / "data"
TXT_DIR = OUT_DIR / "raw_text"
TXT_DIR.mkdir(parents=True, exist_ok=True)

REF_HEADINGS = re.compile(
    r"^\s*(references|bibliography|reference list|literature cited)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Carimbos de download/copyright repetidos em cada página (IEEE Xplore etc.)
STAMP_RE = re.compile(
    r"authorized licensed use|downloaded on .* from ieee xplore|"
    r"restrictions apply|©|\(c\) \d{4} ieee|all rights reserved|"
    r"creative commons|licensee mdpi",
    re.IGNORECASE,
)


def drop_page_furniture(text: str) -> str:
    """Remove carimbos de copyright e linhas idênticas repetidas >=3 vezes
    (cabeçalhos/rodapés de página)."""
    lines = text.split("\n")
    from collections import Counter
    counts = Counter(l.strip() for l in lines if len(l.strip()) >= 10)
    kept = []
    for l in lines:
        s = l.strip()
        if STAMP_RE.search(s):
            continue
        if len(s) >= 10 and counts[s] >= 3:
            continue
        kept.append(l)
    return "\n".join(kept)

YEAR_RE = re.compile(r"[-–—]\s*(19|20)(\d{2})[a-z]?\s*[-–—]")


def parse_filename(name: str):
    """'Autor et al. - 2020 - Titulo.pdf' -> (autores, ano, titulo)."""
    stem = Path(name).stem
    m = YEAR_RE.search(stem)
    if m:
        year = int(m.group(1) + m.group(2))
        authors = stem[: m.start()].strip(" -–—")
        title = stem[m.end():].strip(" -–—")
    else:
        year, authors, title = None, "", stem
    return authors, year, title


def strip_references(text: str) -> str:
    """Corta a secção de referências se o cabeçalho aparecer na metade final."""
    matches = list(REF_HEADINGS.finditer(text))
    if matches:
        last = matches[-1]
        if last.start() > len(text) * 0.5:
            return text[: last.start()]
    return text


def extract(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages)


def main():
    rows = []
    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    print(f"{len(pdfs)} PDFs encontrados")
    for i, pdf in enumerate(pdfs, 1):
        doc_id = f"doc{i:03d}"
        authors, year, title = parse_filename(pdf.name)
        try:
            full = extract(pdf)
        except Exception as e:
            print(f"ERRO em {pdf.name}: {e}")
            rows.append([doc_id, pdf.name, authors, year, title, 0, 0, "ERROR"])
            continue
        body = strip_references(drop_page_furniture(full))
        out = TXT_DIR / f"{doc_id}.txt"
        out.write_text(body, encoding="utf-8")
        rows.append([
            doc_id, pdf.name, authors, year, title,
            len(full.split()), len(body.split()), "OK",
        ])
        if i % 20 == 0:
            print(f"  {i}/{len(pdfs)}...")

    with open(OUT_DIR / "metadata.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "doc_id", "filename", "authors", "year", "title",
            "words_full", "words_no_refs", "status",
        ])
        w.writerows(rows)

    ok = sum(1 for r in rows if r[-1] == "OK")
    no_year = sum(1 for r in rows if r[3] is None)
    print(f"Concluído: {ok}/{len(rows)} OK; {no_year} sem ano no nome do ficheiro")


if __name__ == "__main__":
    sys.exit(main())
