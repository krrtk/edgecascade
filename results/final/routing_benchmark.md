# EdgeCascade Routing & Evaluation Benchmark

**Date:** 2026-09-22T16:56:34.437913
**Judge Provider:** Groq (qwen/qwen3.8-27b)
**Routing Method:** EvidenceGate (Deterministic)
**Evaluation Method:** Batched LLM Semantic Evaluation
**Benchmark Size:** 30 questions

## System Performance

| System | Correct | Partial | Incorrect | API Error | Accuracy (Strict) | RAG Calls | Tier-3 Calls | Judge Calls |
|---|---|---|---|---|---|---|---|---|
| Local | 1 | 1 | 28 | 0 | 3.33% | 0 | 0 | 30 |
| Rag | 6 | 1 | 23 | 0 | 20.00% | 30 | 0 | 30 |
| Remote | 6 | 5 | 19 | 0 | 20.00% | 0 | 30 | 30 |
| Cascade | 7 | 4 | 19 | 0 | 23.33% | 26 | 20 | 30 |

## EdgeCascade Tier Distribution

- **Tier 1 (Local):** 4
- **Tier 2 (RAG):** 6
- **Tier 3 (Remote):** 20
- **Percentage answered without Tier 3:** 33.33%

## Key Comparisons

* **Remote Generation Reduction Vs Always Remote:** 10
* **Rag Call Reduction Vs Always Rag:** 4
* **Remote Calls Per Correct Answer:** 2.86
* **Physical Judge Api Calls:** 30

## Architecture Note
The LLM Judge is an **EVALUATION ORACLE ONLY**. It is NOT used for runtime routing. Runtime routing uses a deterministic `EvidenceGate`.
