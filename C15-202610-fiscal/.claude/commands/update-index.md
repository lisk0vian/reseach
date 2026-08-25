Generate the file context/INDEX.md based on the .md files 
currently in the context/ folder.

For each .md file in context/, add a row to a table 
with the columns: File | Source | Approx Tokens | Description

- "File": name of the .md file inside context/
- "Source": path to the original file it was generated from
- "Approx Tokens": estimate the token count (use wc -w or characters/4)
- "Description": one-line summary of what the file covers, 
  based on its actual content (do not invent)

Do not modify the existing .md files, only generate INDEX.md.
If INDEX.md already exists, overwrite it entirely with the updated info.
