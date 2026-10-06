#!/usr/bin/env bash
# Builds the kickoff report as a LaTeX-composed PDF.
#
#   ./scripts/build-pdf.sh
#
# Output: proposal/arranque.pdf  (plus arranque.tex, because keep-tex is on).
# Rendering the whole site (`quarto render`) also produces the PDF, at
# _site/proposal/arranque.pdf, and adds an "Other Formats" link on the page.
set -e
cd "$(dirname "$0")/.."
quarto render proposal/arranque.qmd --to pdf
echo
echo "PDF:  proposal/arranque.pdf"
echo "TeX:  proposal/arranque.tex"
