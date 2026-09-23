# EdgeCascade

A capability-aware LLM cascade that routes queries between a local instruction model, domain retrieval, and a remote LLM to balance answer quality against unnecessary remote inference.

## Architecture
See the detailed [Architecture Diagram](docs/architecture.md).

EdgeCascade utilizes a three-tier system gated by deterministic evidence logic:
1. **Tier 1 (Local LLM)**: Attempts to answer the query locally. 
2. **Evidence Gate**: Evaluates the Tier 1 answer against retrieved TF-IDF evidence. If it is well-supported (numbers and key entities match), it passes.
3. **Tier 2 (RAG)**: If Tier 1 fails, the system fetches relevant context via TF-IDF retrieval and re-prompts the local model.
4. **Evidence Gate**: Evaluates the Tier 2 RAG answer against the context. If well-supported, it passes.
5. **Tier 3 (Remote LLM)**: If Tier 2 also fails, the query escalates to a powerful remote model to guarantee high quality.

**Runtime Routing Flow:**
```text
Query
  ↓
Local Qwen
  ↓
Evidence Gate
  ↓
TF-IDF RAG
  ↓
Evidence Gate
  ↓
Remote LLM
```

## Evaluation Methodology

EdgeCascade explicitly and permanently separates **runtime routing decisions** from **benchmark evaluation**. 

**Evaluation Flow:**
```text
Candidate answers
      ↓
LLM Evaluation Judge
      ↓
Semantic correctness
```

The LLM Judge is an **EVALUATION ORACLE ONLY**. It is **NOT** used for runtime routing.

## Key Results
Based on a 30-question semantic benchmark using Qwen-0.5B-Instruct and a deterministic Evidence Gate:
Please see the [Final Routing Benchmark Report](results/final/routing_benchmark.md) for the detailed, accurate routing distribution and correctness metrics. The evidence gate enforces strict entity/number checking, leading to highly conservative (and truthful) acceptance rates compared to early heuristic-based prototypes.

## Technologies
- Python, FastAPI
- HuggingFace Transformers (Qwen-0.5B)
- Scikit-learn (TF-IDF Retrieval)
- SQLite (Logging and Evaluation tracking)
- Groq/OpenAI (API Judge & Tier 3 capability - optional but recommended)

## Project Structure
- `api/` — FastAPI application and SQLite database logic.
- `static/` — Simple HTML/JS web interface for the interactive demo.
- `inference/` — Pipeline logic, LLM generation, Evidence gate, RAG.
- `scripts/` — Evaluation, synthetic data generation, and demo scripts.
- `data/` — Evaluation datasets, SQLite database (`edgecascade.db`), and source documents.
- `results/final/` — Final metrics, evaluations, and reports.
- `docs/` — Architecture documentation.

## Running the Application Locally

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. (Optional but recommended) Add an API key to `.env` for the robust API Judge and Tier 3 Remote LLM:
```bash
GROQ_API_KEY=your_key_here
```

3. Start the FastAPI application:
```bash
uvicorn api.main:app --reload
```

4. Navigate to the UI:
Go to `http://localhost:8000/` in your browser.

## Running with Docker

You can easily run the entire application stack using Docker Compose. Note: The Qwen-0.5B model weights will be downloaded to the container on first startup if no explicit model path is provided.

1. Build and start the container:
```bash
docker compose up --build
```
2. The UI is available at `http://localhost:8000/`. The SQLite database is mounted to the `edgecascade-data` volume so it persists across container restarts.

## API Endpoints

- **`GET /health`**
  Returns basic health status.
  ```json
  {"status": "ok", "pipeline_ready": true}
  ```

- **`POST /query`**
  Submit a question to the pipeline.
  ```json
  // Request
  {"domain": "isro", "question": "What is Resourcesat-2 used for?"}
  // Response
  {"request_id": "...", "answer": "...", "tier": "tier2", "rag_used": true, "remote_used": false, "latency_ms": 1500}
  ```

- **`POST /feedback`**
  Submit positive or negative feedback for a given request.
  ```json
  // Request
  {"request_id": "...", "feedback": "positive"}
  ```

- **`GET /metrics`**
  Returns aggregated pipeline metrics (total requests, tier distributions, average latency, feedback counts).

## Known Limitations

**Current routing gate is experimental; evaluation showed over-acceptance of Tier-1 answers.** 
Evidence-based routing refinement is the next research iteration. The system currently accepts highly confident localized responses without fully validating truthfulness, which was identified during the latest semantic benchmark.

- **Small Local Model**: The 0.5B Qwen model lacks parameter count to memorize specific domain facts.
- **Experimental Proof-of-Concept**: This system was built to demonstrate cascading principles; it is not a production-ready application.

## Experimental Custom GPT-2 Work
Early in this project, we explored training a tiny 124M parameter GPT-2 model from scratch using LoRA to see if we could achieve extreme cost reduction. Despite structural coherence, the model lacked the capacity for factual domain QA and was subsequently replaced by the Qwen-0.5B model. This historical code remains in the repository for context.
