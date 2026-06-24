#!/usr/bin/env python3
"""Render the human-readable literature catalog and safe BibTeX stubs.

The CSV registry is authoritative. Stubs intentionally omit unverified authors/DOIs.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CATEGORY_LABELS = {
    "general_lic": "General Learned Image Compression",
    "entropy_modeling": "Entropy Modeling",
    "transformer_lic": "Transformer / SSM / Graph LIC",
    "practical_lic": "Practical and Deployable LIC",
    "rate_perception": "Rate–Distortion–Perception",
    "generative_compression": "Generative Compression",
    "ultralow_generative": "Ultra-Low-Rate Generative Compression",
    "vq_compression": "VQ-Based Compression",
    "text_generative_compression": "Text-Explicit Generative Compression",
    "semantic_text_guided_compression": "Semantic Text-Guided Compression",
    "icm": "Image Coding for Machines",
    "standard_icm": "Standards for Human/Machine Coding",
    "ocr_aware": "OCR-Aware and Text-Preserving Compression",
    "robustness_security": "Robustness and Security",
    "raw_isp": "RAW / ISP-Aware Compression",
    "raw_icm": "RAW Coding for Machines",
}

CATEGORY_ORDER = list(CATEGORY_LABELS)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def escape_bib(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("&", "\\&")
    )


def render_catalog(rows: list[dict[str, str]]) -> str:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)

    out = [
        "# Literature Catalog",
        "",
        "**Snapshot:** 2026-06-24  ",
        "**Source of truth:** `paper_registry.csv`",
        "",
        "This catalog renders every registry entry. Performance numbers are author-reported unless a reproduction record explicitly says otherwise. Cross-paper numerical comparison requires matching reference codec, test protocol, training data and actual-rate accounting.",
        "",
    ]
    ordered = CATEGORY_ORDER + sorted(set(grouped) - set(CATEGORY_ORDER))
    for category in ordered:
        entries = grouped.get(category, [])
        if not entries:
            continue
        out += [f"## {CATEGORY_LABELS.get(category, category.replace('_', ' ').title())}", ""]
        for row in sorted(entries, key=lambda r: (int(r["year"] or 0), r["title"].casefold())):
            out += [
                f"### {row['title']} ({row['year']}, {row['venue']})",
                "",
                f"- **Evidence status:** {row['status']}",
                f"- **Core idea:** {row['key_idea']}",
                f"- **Datasets:** {row['datasets']}",
                f"- **Metrics:** {row['metrics']}",
                f"- **Reported result:** {row['reported_result']}",
                f"- **Difference / caveat:** {row['caveats']}",
                f"- **Implementation:** {row['code_status']}; {row['code_url'] or 'no verified public implementation'}",
                f"- **Primary source:** {row['primary_url']}",
                f"- **Research priority:** {row['priority']}",
                "",
            ]
    return "\n".join(out).rstrip() + "\n"


def render_stubs(rows: list[dict[str, str]]) -> str:
    blocks: list[str] = []
    for row in rows:
        blocks.append(
            "\n".join(
                [
                    f"@misc{{registry_{row['id']},",
                    f"  title = {{{escape_bib(row['title'])}}},",
                    f"  year = {{{row['year']}}},",
                    f"  howpublished = {{{escape_bib(row['venue'])}}},",
                    f"  url = {{{row['primary_url']}}},",
                    "  note = {Registry stub; author/DOI/venue metadata must be verified before publication. "
                    f"Evidence status: {escape_bib(row['status'])}}}",
                    "}",
                ]
            )
        )
    return "\n\n".join(blocks) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail if rendered files differ")
    args = parser.parse_args()

    registry = ROOT / "literature/paper_registry.csv"
    rows = load_rows(registry)
    expected = {
        ROOT / "literature/catalog.md": render_catalog(rows),
        ROOT / "literature/papers_registry_stubs.bib": render_stubs(rows),
    }
    changed: list[str] = []
    for path, content in expected.items():
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current != content:
            changed.append(str(path.relative_to(ROOT)))
            if not args.check:
                path.write_text(content, encoding="utf-8")
    if args.check and changed:
        print("Out-of-date rendered files:", ", ".join(changed))
        return 1
    print(f"Rendered {len(rows)} literature records" if not args.check else f"OK: {len(rows)} records in sync")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
