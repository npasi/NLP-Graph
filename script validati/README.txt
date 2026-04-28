SCRIPT VALIDATI

Questa cartella contiene SOLO gli script "stabili" già testati.

1) Split in capitoli (solo splitter, nessun parsing):
   - step1_split_only.py
   - chapter_splitter.py

Esempio (Christmas Carol già pulito in data/cleaned/46.txt):

  .\.venv\Scripts\python -X utf8 -m src.step1_split_only --input "data\cleaned\46.txt" --book-id "christmas_carol" --debug

Output:
  data/books/christmas_carol/chapters/chapters.json
  data/books/christmas_carol/chapters/chapter_000.txt ...
  data/books/christmas_carol/chapters/split_debug.json

