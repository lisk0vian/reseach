# Journal of Big Data — Submission Requirements & Guidelines

**Publication**: Journal of Big Data (Springer, SpringerOpen)  
**ISSN**: 2196-1115  
**URL**: https://journalofbigdata.springeropen.com/  
**Official Guidelines**: https://journalofbigdata.springeropen.com/submission-guidelines

---

## MANUSCRIPT FORMATTING REQUIREMENTS

### Language & General Rules
- **Language**: Concise English only
- **File Format**: Editable files required (Word .docx or LaTeX .tex)
- **Special Characters**: Must be embedded in text (no external font dependencies)
- **Avoid**: Footnotes for references/citations; use superscript numbers instead

### Abstract & Keywords
- **Abstract Length**: 150–250 words
  - Must briefly summarize aim, findings, or purpose
  - Minimize abbreviations
  - **Do NOT cite references** in abstract
- **Keywords**: 3–10 keywords representing main content
  - Use lowercase except for proper nouns and acronyms

### Article Structure (Research Articles)

Recommended section order (typical for methodology papers):
1. **Title**
2. **Author Information** (with affiliations, ORCID, email)
3. **Abstract** (150–250 words)
4. **Keywords** (3–10)
5. **Introduction**
6. **Materials and Methods** (or "Methodology")
7. **Results**
8. **Discussion** (or "Results and Discussion")
9. **Limitations** (if applicable)
10. **Conclusions**
11. **Acknowledgments** (if applicable)
12. **Competing Interests** (mandatory)
13. **Funding** (mandatory if funded)
14. **Data Availability** (if applicable)
15. **References**
16. **Appendices** (if applicable)

### Competing Interests & Funding Declaration
**MANDATORY** for all articles.

**Competing Interests**:
- Declare all financial AND non-financial competing interests
- If none: "The authors declare that they have no competing interests."

**Funding**:
- Declare all sources of funding
- If funder has specific role (conceptualization, design, data collection, analysis, decision to publish, preparation): declare this explicitly
- If no funding: "No funding was received for this research."

### Data Availability & Citation
- **MANDATORY**: Cite any publicly available data on which conclusions rely
- Authors should deposit data in recognized repositories (e.g., GitHub, Zenodo, OSF, Figshare)
- If data cannot be publicly shared, state reasons clearly

### Word Count & Length Limits
- **NO STRICT WORD LIMIT** for most article types
- Length and quantity of supporting data is NOT restricted
- Advantage: methodological papers with extensive validation (ablation studies, SHAP analysis, etc.) are encouraged

---

## FIGURES & TABLES

### Figures
- **Numbering**: In order of first mention in text
- **Titles**: Maximum 15 words
- **Legends/Captions**: Maximum 300 words
- **Format**: 
  - Submit as SEPARATE files from manuscript
  - Multi-panel figures → submit as single composite file
  - Supported: PDF, EPS, PNG, JPG (high resolution: ≥300 dpi)
  - Include descriptive legend in main manuscript, NOT in graphic file
- **Placement**: Figures go at end or inline where first referenced
- **File Naming**: e.g., `Figure1.pdf`, `Figure2a.png`, etc.

### Tables
- **Numbering**: In order of first mention in text
- **Titles**: Maximum 15 words
- **Legends**: Maximum 300 words
- **Format**:
  - **Must be in main manuscript file** (NOT as separate images)
  - Use native table editor (Word Tables or LaTeX tabular environment)
- **Placement**:
  - Tables ≤ 1 A4/Letter page: place inline at appropriate location
  - Tables > 1 A4/Letter page: place at end of document
- **Avoid**: Merged cells, complex formatting that doesn't survive PDF conversion

---

## REFERENCES & CITATIONS

### Citation Style
- **Primary Style**: **Vancouver Reference Style** (numbered)
- **Secondary Styles Available**: 
  - Harvard (author-date)
  - APA (author-date)
  - Check journal submission portal for available BibTeX styles

### Formatting Rules

**For Vancouver Style (default)**:
- Citations in text: Superscript numbers in order of appearance
- Example in text: "...as demonstrated [1], with additional evidence [2,3]."
- Reference list: Numbered [1], [2], [3], etc., in order of appearance
- Format: `[#] Authors. Title. Journal Year;Volume(Issue):Pages.`

**Example Reference List**:
```
[1] Smith JD, Johnson AB, Williams CD. Machine learning for judicial prediction. 
    Journal of Big Data. 2024;15(4):142-158.

[2] Brown EF, Davis GH, Miller IJ, et al. Prosecutorial analytics and operational 
    risk assessment. Justice Systems Review. 2023;12(7):45-61.

[3] Taylor KL. Open data governance in Latin America. Government Information 
    Quarterly. 2024;41(2):101234.
```

**Important Notes**:
- **No URLs in text**: All web links must be given a reference number and placed in reference list
- **No footnotes for citations**: Use superscript numbers only
- **Abbreviate journal names** according to standard ISO 4 abbreviations
- **Include DOIs** when available (format: `https://doi.org/10.xxxx/xxxxx`)

### Bibliography Management
- Recommended tools: BibTeX (.bib files) with Springer .bst styles (included in templates)
- Available .bst files in Springer template:
  - `sn-vancouver.bst` (DEFAULT for most Springer journals)
  - `sn-aps.bst` (APS/Physics style)
  - `sn-nature.bst` (Nature style)
  - `sn-chicago.bst` (Chicago style)
  - `sn-apacite.bst` (APA-extended style)
  - `sn-basic.bst` (Basic numbered style)

---

## SUBMISSION CHECKLIST

- [ ] Manuscript in concise English
- [ ] Abstract 150–250 words (no references cited)
- [ ] Keywords: 3–10
- [ ] All special characters embedded
- [ ] Editable files only (no PDFs as main manuscript)
- [ ] Figures numbered in order, with separate files + captions in manuscript
- [ ] Tables in main manuscript file, properly formatted
- [ ] References in Vancouver style (or selected style), numbered in text order
- [ ] No URLs in text (all in reference list with numbers)
- [ ] Competing Interests section included and completed
- [ ] Funding section included and completed
- [ ] Data Availability statement included (with public repository links if applicable)
- [ ] All figures/tables cited in text
- [ ] No submission to other journals simultaneously

---

## TEMPLATES & RESOURCES

### Available Templates

#### 1. LaTeX Template (INSTALLED)
- **Location**: `springer-nature-latex-template-master/`
- **Main Files**:
  - `sn-article.tex` — Complete example article
  - `sn-jnl.cls` — Document class (do NOT modify)
  - `sn-vancouver.bst` — Bibliography style (use for Journal of Big Data)
  - `sn-bibliography.bib` — Example bibliography file
  - `user-manual.pdf` — Comprehensive usage guide
  - `sn-article.pdf` — Rendered PDF example

**Quick Start LaTeX**:
1. Copy `sn-article.tex` as your main file
2. Use `\documentclass[twocol,11pt]{sn-jnl}` (default)
3. Use `\bibstyle{sn-vancouver}` for references
4. Compile with: `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`

#### 2. Word Template
- **Location**: Available from official Springer page
- **Official Download**: 
  - https://www.springer.com/gp/authors-editors/journal-author/word-template-zip-154-kb-/22044
  - Or via Springer Support: https://support.springernature.com/en/support/solutions/articles/6000081241-templates-and-style-files-for-journal-article-preparation
- **Note**: Download manually or contact journal directly for DOCX file
- **Format**: Microsoft Word 2010 or later (.docx)

#### 3. Online Editor (Overleaf)
- **LaTeX Online**: https://www.overleaf.com/latex/templates/springer-nature-latex-template/myxmhdsbzkyd
- Advantage: No local LaTeX installation needed
- Supports real-time collaboration
- One-click submission to SpringerOpen (when available)

---

## SUBMISSION PROCESS

1. **Prepare manuscript** using template (LaTeX or Word)
2. **Upload files**:
   - Main manuscript (editable: .tex or .docx)
   - Figure files (separate files for each figure)
   - Supporting data (if applicable; additional files welcome)
   - Metadata (author, affiliations, keywords)
3. **Declare conflicts**: Competing interests, funding
4. **Data statement**: Confirm data availability / public repository
5. **Submit** via SpringerOpen Editorial Manager
6. **Review timeline**: Expected ~13 weeks until decision

---

## IMPORTANT NOTES FOR YOUR MANUSCRIPT

For **prosecutorial congestion prediction** paper:
- ✓ Your extensive methodology section (6 feature selection methods, temporal validation, ablation analysis, SHAP explanations) → fits perfectly with Journal of Big Data's appreciation for rigorous methodological papers
- ✓ Your open-data focus (MPFN dataset) → aligns with journal's support for reproducible research with public data
- ✓ No strict word limit → advantage for detailed validation sections
- ✓ Emphasis explainability + robustness → matches journal's ML best-practices standards

**Template Recommendation**: Use **LaTeX** with `sn-vancouver.bst` for full control over formatting and citations.

---

## REFERENCES
- Official Journal Guidelines: https://journalofbigdata.springeropen.com/submission-guidelines
- Springer Support: https://support.springernature.com
- LaTeX on Overleaf: https://www.overleaf.com/latex/templates/springer-nature-latex-template/myxmhdsbzkyd

**Last Updated**: August 24, 2026
