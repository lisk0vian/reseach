---
name: build-latex
description: Build and compile LaTeX manuscript to PDF, automatically copying to papers/renders/
tools: [Bash, PowerShell]
---

# Build LaTeX Manuscript

Compiles the Journal of Big Data LaTeX manuscript and automatically copies the generated PDF to `papers/renders/`.

## Usage

### On Mac/Linux/WSL:
```bash
cd papers/drafts/latex
make
```

### On Windows (PowerShell):
```powershell
cd papers/drafts/latex
.\build-latex.ps1
```

## What It Does

1. ✅ Runs pdflatex + bibtex + cross-reference passes
2. ✅ Generates PDF from `main-simple.tex`
3. ✅ Copies result to `papers/renders/C15-202610[fiscal].latex.en.pdf`
4. ✅ Cleans up auxiliary files

## Output

**Generated PDF:** `papers/renders/C15-202610[fiscal].latex.en.pdf`

## Documentation

For detailed help:
- **Quick start:** `papers/drafts/latex/QUICKSTART.md`
- **Full guide:** `papers/drafts/latex/BUILD.md`
- **Skill docs:** `papers/drafts/latex/SKILL-DOCUMENTATION.md`

## Available Commands

### Using Make (Mac/Linux/WSL):
```bash
make                 # Build main-simple (default)
make all             # Clean + build + copy
make clean           # Remove auxiliary files
make help            # Show all targets
```

### Using PowerShell (Windows):
```powershell
.\build-latex.ps1                      # Default build
.\build-latex.ps1 -TexFile main        # Build Springer version
.\build-latex.ps1 -Clean               # Clean before building
```

### Using Bash Script:
```bash
./build-latex.sh                       # Default build
./build-latex.sh main                  # Build Springer version
```

## After Building

1. Check the PDF: `papers/renders/C15-202610[fiscal].latex.en.pdf`
2. Review for errors or formatting issues
3. When ready, submit to Journal of Big Data

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `make: command not found` | Use PowerShell script instead: `.\build-latex.ps1` |
| `pdflatex: command not found` | Install TeX Live or MiKTeX (see BUILD.md) |
| PDF looks wrong | Run full rebuild: `make clean && make all` |
| Very slow | Normal on first build; subsequent builds are fast |

## More Info

See `papers/drafts/latex/BUILD.md` for comprehensive documentation including:
- CI/CD setup (GitHub Actions)
- Git hooks integration
- IDE integration (VS Code, Overleaf)
- Performance optimization
- Advanced usage
