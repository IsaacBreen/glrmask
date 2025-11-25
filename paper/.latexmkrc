# Latexmk configuration

# Use pdflatex
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';

# Use bibtex for bibliography
$bibtex_use = 2;

# Clean up extra files
$clean_ext = 'bbl fls nav out snm synctex.gz vrb run.xml';

# Output directory (optional, uncomment to use)
# $out_dir = 'build';
