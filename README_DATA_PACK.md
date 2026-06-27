# Document Navigator: A Transparent RAG Assistant – Sample Data Pack

This pack contains a **synthetic, test-only** set of PDFs plus a small evaluation set.
It is designed for a local PDF question-answering system that shows **why** it answered:
- Retrieval traces (top-k sources + similarity scores)
- Cited answers using a consistent format
- Lightweight evaluation (precision@k + examples)

## Contents
- pdfs/ : 10 synthetic PDFs (1 page each)
- eval_set.csv : 15 questions with gold citations and key phrases
- retrieval_report_template.md : report template
- README_DATA_PACK.md : this file

## Citation Format
Use `[filename:page]` (all PDFs are page 1 in this dataset).

## Suggested Retrieval Evaluation
- Build a vector index over chunked PDF text.
- For each eval question, retrieve top-k chunks and compute precision@k
  by checking whether the gold citation’s file appears in top-k.

## Suggested Answer Evaluation
- Use key-phrase string match for a lightweight score.
- Include manual checks for citation correctness and “weak evidence” behavior.
