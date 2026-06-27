# retrieval_report.md – Document Navigator

## Setup
- PDFs: 10
- Chunk size : 400 characters
- Overlap    : 80 characters
- Embeddings model : all-MiniLM-L6-v2  (sentence-transformers)
- Vector index     : FAISS IndexFlatIP  (cosine similarity)

## Metrics
- precision@3           : 0.3778  (37.8 %)
- precision@5           : 0.2667  (26.7 %)
- top-1 accuracy          : 0.8667  (86.7 %)
- key-phrase match rate   : 0.6667  (66.7 %)

## Examples (Good)
- Q: What is the standard delivery timeline?
  - Retrieved : policy_shipping_returns.pdf (score=0.4200)
  - Answer    : [Confidence: 0.42 – moderate.  The answer below is based on the best available match; please verify with the cited source.]

Standard delivery takes 3–6 busines

- Q: What is the return window for most products?
  - Retrieved : policy_shipping_returns.pdf (score=0.3415)
  - Answer    : [Confidence: 0.34 – moderate.  The answer below is based on the best available match; please verify with the cited source.]

Most products can be returned withi

- Q: How long do refunds typically take after quality check?
  - Retrieved : policy_shipping_returns.pdf (score=0.5940)
  - Answer    : Refunds are processed within 3–7 business days after quality check. [policy_shipping_returns.pdf:1]

## Examples (Failures)
- Q: What is hybrid retrieval?
  - Retrieved : guide_evaluation_metrics.pdf  (expected guide_vector_search.pdf)
  - Why it failed : Top-1 source did not match gold.
  - Fix attempted : Tune chunk size / try hybrid retrieval.

- Q: What should you do if evidence is weak?
  - Retrieved : guide_support_escalation.pdf  (expected guide_rag_basics.pdf)
  - Why it failed : Top-1 source did not match gold.
  - Fix attempted : Tune chunk size / try hybrid retrieval.
