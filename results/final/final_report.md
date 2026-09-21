# EdgeCascade — Final Report

## 1. Problem

A common challenge in deploying small local LLMs on edge devices is that they may not reliably answer every complex, domain-specific query. While they can handle simple or general questions efficiently, they are prone to confident hallucinations when faced with detailed technical questions that exceed their capacity. Running a large remote model for every query solves quality but introduces unacceptable latency and cost.

## 2. Architecture

EdgeCascade uses a multi-tier routing architecture to balance answer quality against inference costs:

```
User
 ↓
Practical Local LLM (Tier 1)
 ↓
Semantic Judge
 ↓
PASS → Answer
FAIL
 ↓
TF-IDF Retrieval
 ↓
Local LLM + Context (Tier 2)
 ↓
Semantic Judge
 ↓
PASS → Answer
FAIL
 ↓
Remote LLM (Tier 3)
```

## 3. Custom GPT-2 From-Scratch Component

Early in the project, we experimented with training a custom GPT-2 124M model from scratch:
- **Architecture**: Custom Transformer implementation based on the GPT-2 124M architecture.
- **Pretrained Weights**: Loaded from HuggingFace to baseline capability.
- **Q/V LoRA**: Implemented parameter-efficient fine-tuning via LoRA on Q and V attention matrices.
- **SFT Experiment**: We transitioned the objective from Causal Continuation to Supervised Fine-Tuning (SFT) for Q&A.
- **Capability Probe**: We discovered that despite the coherent structure, the 124M model's capacity was simply too constrained to produce factual domain answers. When evaluated on unseen questions, the generated outputs remained structurally coherent but factually nonsensical (e.g., hallucinating generic filler).
- **Result**: Due to severe capacity constraints, it was not retained as the practical local QA model for the final cascade, as a stronger base model is necessary for Tier 1 to successfully answer a significant percentage of queries.

## 4. Practical Local Model

For the practical cascade, we integrated **Qwen/Qwen2.5-0.5B-Instruct**.
This model was selected because it is highly capable for its size (0.5B parameters), operates efficiently on edge devices (including CPU), and inherently follows instructions better than our custom 124M model. However, despite being more coherent, it still hallucinates factual domain knowledge heavily without context. It is coherent, but its unassisted accuracy on our specific benchmark is limited.

## 5. RAG

To rescue queries where the local model fails, we implemented a Tier 2 Retrieval-Augmented Generation (RAG) system:
- **TF-IDF Retrieval**: A lightweight keyword-based TF-IDF retriever.
- **Domain-specific corpus**: We segmented domain documents into retrieval chunks.
- **Local Generation with Context**: When Tier 1 fails, the system fetches the top chunks and re-prompts the local model with the retrieved context as evidence, effectively grounding the local generation.

## 6. Routing Judge

**Benchmark routing judge: heuristic fallback**

The final reported benchmark was run using a local heuristic fallback judge (`SemanticRoutingJudge`). Because a remote API key was not available in the environment during the final benchmark run, the system fell back to a heuristic evaluator that measures semantic overlap and generation confidence. This is **not** an API/LLM-based judge.

## 7. Experimental Setup

- **Benchmark**: 30 frozen questions spanning two domains.
- **Domains**: ISRO (Space/Technical) and DPDPA (Legal/Privacy).
- **Categories**: Knowledge-gap queries and Reasoning queries.

## 8. Results

*Metrics below reflect the final benchmark run using the practical Qwen 0.5B system and the heuristic fallback judge.*

| System | Final correctness | Tier 1 | Tier 2 | Tier 3 | RAG calls | Remote calls | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Old GPT-2 System** | ~86.7% | 0 | 0 | 30 | 0% | 100% | N/A |
| **Practical Qwen System** | 93.3%* | 29 | 1 | 0 | 3.3% | 0.0% | ~7.9s |

*\*Note: "Final correctness" measures keyword/overlap correctness against the ground truth. As detailed in the Failure Analysis, the semantic factual quality of many Tier 1 answers was actually poor despite passing the heuristic judge.*

## 9. Failure Analysis

### RAG Rescue (Representative Success)
- **Question:** What is the OBSSR capacity?
- **Tier 1 Failure:** The local model hallucinated about "Obstacle Satellites" and the judge appropriately FAILED it.
- **RAG Rescue:** The system retrieved relevant chunks, and the local model then successfully generated: "The OBSSR capacity is 120 Gb..."
- **Result:** Correctly rescued at Tier 2. (Quality: GOOD)

### Local Hallucination (Heuristic Judge Limitation)
- **Question:** What is the swath of Cartosat-1 in stereo mode?
- **Tier 1 Answer:** "The swath of Cartosat-1 in stereo mode covers approximately 34,000 square kilometers..."
- **Result:** The ground truth is 26 km. The local model completely hallucinated the answer. However, because the benchmark used the heuristic fallback judge, it passed this confident hallucination at Tier 1, skipping RAG entirely.

## 10. Limitations

- **Small Local Model**: The 0.5B Qwen model lacks the parameter count to memorize specific domain facts.
- **CPU Latency**: Inference on CPU averages ~8 seconds per query.
- **Small Benchmark**: Evaluated on a frozen 30-question set; not representative of large-scale production.
- **Judge Limitations**: The final benchmark relied on a heuristic fallback judge rather than a robust API judge.
- **No Production Guarantee**: This is an experimental proof-of-concept. It does not provide production-grade hallucination prevention.

## 11. Conclusion

EdgeCascade demonstrates that a multi-tier routing architecture is technically viable. The cascade mechanism works in principle—demonstrated by successful RAG rescues when the judge correctly fails weak answers. However, the experiment also explicitly demonstrates that **cascading is entirely dependent on the quality of the routing judge**. When falling back to a heuristic judge, the system prematurely accepts confident hallucinations at Tier 1, preventing necessary escalation. Therefore, EdgeCascade does not prove that cascading *always* reduces cost or latency safely; rather, it highlights that a cascade can only reduce remote calls safely if paired with an exceptionally robust (often LLM-based) evaluator.
