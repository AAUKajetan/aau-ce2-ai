# Tools

Utility scripts used to build the study materials (not study material itself).

- `clean_pdf_md.py` — cleans pymupdf-extracted markdown (removes extraction
  artifacts). Used when converting lecture/exercise PDFs into the `exam_prep/*.md`
  notes. Usage: `python clean_pdf_md.py path/to/file.md` → writes `<file>_clean.md`.
