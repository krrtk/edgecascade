# EdgeCascade

A capability-aware LLM cascade that routes queries between a local instruction model, domain retrieval, and a remote LLM to balance answer quality against unnecessary remote inference.

## Architecture
See the detailed [Architecture Diagram](docs/architecture.md).

EdgeCascade utilizes a three-tier system:
1. **Tier 1 (Local LLM)**: Attempts to answer the query locally. A semantic judge evaluates the answer.
2. **Tier 2 (RAG)**: If Tier 1 fails, the system fetches relevant context via TF-IDF retrieval and re-prompts the local model.
3. **Tier 3 (Remote LLM)**: If Tier 2 also fails, the query escalates to a powerful remote model to guarantee high quality.

## Key Results
Based on a 30-question benchmark using Qwen-0.5B-Instruct and a local heuristic fallback judge:
- **Final correct answers (overlap-based)**: 93.3%
- **Remote calls**: 0%
- **RAG rescue rate**: 100% of invoked rescues succeeded.

*(Note: The current heuristic fallback judge is lenient. An API-based judge is required for production-grade routing. Please see the [Final Report](results/final/final_report.md) for a detailed failure analysis.)*

## Technologies
- Python, FastAPI
- HuggingFace Transformers (Qwen-0.5B)
- Scikit-learn (TF-IDF Retrieval)
- SQLite (Logging and Evaluation tracking)
- Groq/OpenAI (API Judge & Tier 3 capability - optional but recommended)

## Project Structure
- `inference/` — Pipeline logic, LLM generation, Judge routing, RAG.
- `scripts/` — Evaluation, synthetic data generation, and demo scripts.
- `data/` — Evaluation datasets and source documents.
- `results/final/` — Final metrics, evaluations, and reports.
- `docs/` — Architecture documentation.

## Quickstart

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. (Optional but recommended) Add an API key to `.env` for the robust API Judge and Tier 3 Remote LLM:
```bash
GROQ_API_KEY=your_key_here
# or
OPENAI_API_KEY=your_key_here
```

3. Run the interactive demo:
```bash
python scripts/demo.py
```

## Example Query

### Representative RAG Rescue
**User:** What is the OBSSR capacity?
1. **Tier 1 (Local):** Fails to answer accurately (Hallucinates). Judge rejects.
2. **Tier 2 (RAG):** Context retrieved. Local model generates: *"The OBSSR capacity is 120 Gb."* Judge passes.
3. **Final Result:** High-quality answer returned purely locally.

## Limitations

- **Small Local Model**: The 0.5B Qwen model lacks parameter count to memorize specific domain facts.
- **Judge Dependence**: EdgeCascade's safety entirely relies on its judge. Without an API judge (i.e. using the heuristic fallback), it is prone to prematurely accepting confident hallucinations at Tier 1.
- **Experimental Proof-of-Concept**: This system was built to demonstrate cascading principles; it is not a production-ready application.

## Experimental Custom GPT-2 Work
Early in this project, we explored training a tiny 124M parameter GPT-2 model from scratch using LoRA to see if we could achieve extreme cost reduction. Despite structural coherence, the model lacked the capacity for factual domain QA and was subsequently replaced by the Qwen-0.5B model. This historical code remains in the repository for context.
