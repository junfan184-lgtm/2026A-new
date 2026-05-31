$ErrorActionPreference = "Stop"

if (Get-Command latexmk -ErrorAction SilentlyContinue) {
    latexmk -xelatex -interaction=nonstopmode main.tex
} else {
    xelatex -interaction=nonstopmode main.tex
    biber main
    xelatex -interaction=nonstopmode main.tex
    xelatex -interaction=nonstopmode main.tex
}

Write-Host "Build finished. Rename main.pdf as <18位唯一队号>_A.pdf before submission."
