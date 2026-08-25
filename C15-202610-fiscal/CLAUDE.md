# CLAUDE.md

## Project context
Before exploring `data/` or `papers/drafts/` for background information, 
always check `context/` first.

The `context/` folder contains markdown versions of key project documents, 
converted with markitdown and pandoc to reduce token usage. Use these 
files instead of opening the original source documents (.xlsx, .docx, .doc) 
whenever you need background, definitions, or content from those sources.

- `context/INDEX.md` lists every file in context/, its original source, 
  an approximate token count, and a one-line description. Read INDEX.md 
  first to decide which file(s) are actually relevant before opening them.
- Raw CSVs in `data/` are used directly by notebooks and do NOT have a 
  markdown version in context/ — read them normally when working with 
  notebooks/experiments.ipynb.
- Do not answer questions about project background, methodology, or the 
  paper drafts by exploring the whole repo — check context/INDEX.md first.