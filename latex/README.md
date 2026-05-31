# A题 LaTeX 论文目录

本目录用于撰写 A 题正式论文，格式依据：

`../../赛题A/论文写作规范及团队编号查询/1. 河北省研究生数学建模竞赛论文格式规范.docx`

## 文件说明

- `main.tex`：论文主文件，包含摘要页、正文和参考文献。
- `references.bib`：参考文献数据库。
- `build.ps1`：Windows PowerShell 编译脚本。

## 编译方式

建议使用 XeLaTeX：

```powershell
cd C:\Users\ADMIN\Desktop\数模\A题工作区\latex
.\build.ps1
```

如果编辑器默认使用 `pdflatex`，建议将编译方式改为 `xelatex` 或 `latexmk (xelatex)`。本目录已提供 `.latexmkrc` 和 `latexmkrc`，直接运行 `latexmk main.tex` 会优先使用 XeLaTeX。上传 Overleaf 后，也建议在 Menu 中确认 Compiler 为 `XeLaTeX`。

如本机没有 `latexmk`，可手动执行：

```powershell
xelatex main.tex
biber main
xelatex main.tex
xelatex main.tex
```

## 提交命名

最终 PDF 按竞赛要求命名：

```text
18位唯一队号_A.pdf
```
