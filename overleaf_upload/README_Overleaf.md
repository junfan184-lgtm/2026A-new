# Overleaf 编译说明

请在 Overleaf 中使用本目录作为项目根目录上传，`main.tex` 必须位于项目根目录。

## 必须设置

1. 点击左上角 `Menu`。
2. 将 `Compiler` 设置为 `XeLaTeX`。
3. 将 `Main document` 设置为 `main.tex`。
4. 点击 `Recompile from scratch` 后重新编译。

如果日志第一行仍显示 `preloaded format=pdflatex`，说明 Overleaf 仍在使用 pdfLaTeX，此时中文字体会报错。

## 文件清单

- `main.tex`
- `references.bib`
- `latexmkrc` / `.latexmkrc`
- `figures/overview.png`
- `figures/pressure_strategy.png`
- `figures/robustness_rmse.png`
