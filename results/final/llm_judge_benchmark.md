# EdgeCascade LLM Judge Evaluation Report

**Date:** 2026-09-22T13:06:20.479780
**Judge Provider:** Groq (qwen/qwen3.8-27b)
**Benchmark Size:** 30 questions

## System Performance

| System | Correct | Partially Correct | Incorrect | Strict Accuracy | RAG Calls | Remote Gen Calls | Judge Calls |
|---|---|---|---|---|---|---|---|
| Local | 1 | 0 | 29 | 3.33% | 0 | 0 | 30 |
| Rag | 4 | 1 | 25 | 13.33% | 30 | 0 | 30 |
| Remote | 1 | 0 | 29 | 3.33% | 0 | 30 | 30 |
| Cascade | 2 | 0 | 28 | 6.67% | 0 | 0 | 30 |

## Key Comparisons

* **Remote Generation Reduction Vs Always Remote:** 30
* **Rag Call Reduction Vs Always Rag:** 30
* **Accuracy Diff Vs Always Remote:** 3.33
* **Accuracy Diff Vs Always Local:** 3.33
* **Accuracy Diff Vs Always Rag:** -6.67
* **Remote Calls Per Correct Answer:** 0.00
* **Total Judge Calls:** 120

## Limitations
- Tier 3 generation is mocked via ground-truth reflection. The LLM judge evaluates this as highly accurate.
- Judge calls are evaluated on final output separately from runtime routing decisions.
