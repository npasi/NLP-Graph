# NLP-Graph – Usage Guide

This repository provides an end-to-end pipeline to transform a raw book (.txt)
into a structured character graph using BookNLP and a custom Character Identity Layer.

---

## 🚀 Quick Start (End-to-End)

Run the full pipeline with a single command:

```bash
python -m src.run_full_pipeline --input data/raw/book.txt --book-id book
