# Paper: Grammar-Constrained Decoding

LaTeX source for the research paper.

## Structure

- `main.tex` — Main document
- `references.bib` — Bibliography
- `figures/` — Figures and diagrams
- `sections/` — Individual sections (optional, for modular editing)

## Building

### With latexmk (recommended)

```bash
cd paper
latexmk -pdf main.tex
```

### Manual build

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

### Continuous build (watch mode)

```bash
latexmk -pdf -pvc main.tex
```

## Clean build artifacts

```bash
latexmk -C
```

## Notes

- Use `\todo{...}` for TODO markers (shows in red)
- Use `\note{...}` for notes (shows in blue)
- Place figures in `figures/` directory
- Update `references.bib` with citations

## Target Venues

- ACL / EMNLP (NLP focus)
- NeurIPS / ICML (ML focus)
- ICLR (ML + systems)

Choose based on paper angle (theory vs. systems vs. applications).
