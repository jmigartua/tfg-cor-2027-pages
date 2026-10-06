#!/usr/bin/env bash
# Rebuilds every figure from its pgfplots source.
# Requires: pdflatex with pgfplots, and pdftocairo (poppler) for the SVG step.
set -e
cd "$(dirname "$0")/../figures"
for f in f1_cor_vs_h f2_rise f3_sens f4_prec; do
  pdflatex -interaction=nonstopmode -halt-on-error "$f.tex"
  pdftocairo -svg "$f.pdf" "$f.svg"
done
rm -f *.aux *.log
echo "figures rebuilt"
