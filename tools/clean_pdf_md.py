#!/usr/bin/env python3
"""
PDF markdown cleaner - removes extraction artifacts from pymupdf output.

Usage:
  python clean_pdf_md.py FromAAU/ce2-lecture_1-intro-big-data-hadoop.md
  python clean_pdf_md.py FromAAU/*.md

Outputs:
  <filename>_clean.md   - cleaned version (same dir)
"""

import re
import sys
from pathlib import Path


def clean(text: str) -> str:
    """Remove PDF extraction artifacts. No knowledge added, only noise removed."""

    # Remove PAGE markers like "P A G E\n1 0" or "PAGE\n67" or standalone page numbers
    text = re.sub(r'\n*P\s*A\s*G\s*E\s*\n\s*\d[\d ]*\s*\n*', '\n\n', text)
    # Also catch standalone "55\n" style page numbers at line start
    text = re.sub(r'^\d{1,3}\n', '', text, flags=re.MULTILINE)

    # Collapse spaced-out decorative text like "T h e  E l e m e n t s  o f  D a t a"
    def despaced(m):
        collapsed = re.sub(r'(?<=\S) (?=\S)', '', m.group(0))
        return collapsed if len(collapsed) > 3 else m.group(0)

    text = re.sub(r'^[A-Za-z](?: [A-Za-z]){4,}.*$', despaced, text, flags=re.MULTILINE)

    # Remove repeated Agenda blocks (keep only the first)
    agenda_block = re.compile(
        r'Agenda\n'
        r'Course introduction\n'
        r'Objectives, program, and goals\n'
        r'What is Big Data\?\n'
        r'Characteristics of Big Data \(5Vs\)\n'
        r'Volume, Velocity, Variety, Veracity, Value\n'
        r'Problem with Big Data\n'
        r'Apache Hadoop\s*',
        re.MULTILINE
    )
    matches = list(agenda_block.finditer(text))
    for m in reversed(matches[1:]):
        text = text[:m.start()] + text[m.end():]

    # Compact source references
    text = re.sub(r'\nSource: (https?://\S+)\n', r'\n', text)
    text = re.sub(r'\nSource: ([^\n]+)\n', r'\n', text)

    # Remove lines that are purely bullet decorations (• alone)
    text = re.sub(r'^\s*[•·]\s*$', '', text, flags=re.MULTILINE)

    # Collapse 3+ blank lines to 1
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove lines that are just whitespace
    text = re.sub(r'\n[ \t]+\n', '\n\n', text)

    # Strip trailing whitespace
    text = '\n'.join(line.rstrip() for line in text.split('\n'))

    return text.strip() + '\n'


def main():
    if len(sys.argv) < 2:
        print("Usage: python clean_pdf_md.py <input.md> [<input2.md> ...]")
        sys.exit(1)

    for input_path in sys.argv[1:]:
        path = Path(input_path)
        if not path.exists():
            print(f"[ERROR] File not found: {input_path}")
            continue

        raw_text = path.read_text(encoding='utf-8')
        clean_text = clean(raw_text)

        out_path = path.with_stem(path.stem + '_clean')
        out_path.write_text(clean_text, encoding='utf-8')

        reduction = (1 - len(clean_text) / len(raw_text)) * 100
        print(f"{path.name}: {len(raw_text.splitlines())} -> {len(clean_text.splitlines())} lines ({reduction:.0f}% smaller) -> {out_path.name}")


if __name__ == "__main__":
    main()
