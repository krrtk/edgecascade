# EdgeCascade Architecture

EdgeCascade routes queries through an escalating pipeline to balance inference cost, latency, and answer quality.

## System Pipeline

```mermaid
flowchart TD
    User([User Query]) --> API[FastAPI Endpoint]
    API --> DomainSelect{Domain Selection}
    
    DomainSelect --> ISRO[ISRO Corpus]
    DomainSelect --> DPDPA[DPDPA Corpus]
    
    ISRO --> Tier1
    DPDPA --> Tier1
    
    subgraph Tier 1: Local
        Tier1[Practical Local Model\nQwen-0.5B]
        Judge1{Semantic Judge}
        Tier1 --> Judge1
    end
    
    Judge1 -- PASS --> FinalAnswer([Final Answer])
    Judge1 -- FAIL --> Tier2
    
    subgraph Tier 2: RAG
        Tier2[TF-IDF Retriever]
        Tier2 --> LocalContext[Local Model + Context]
        Judge2{Semantic Judge}
        LocalContext --> Judge2
    end
    
    Judge2 -- PASS --> FinalAnswer
    Judge2 -- FAIL --> Tier3
    
    subgraph Tier 3: Remote
        Tier3[Remote LLM API]
    end
    
    Tier3 --> FinalAnswer
    
    FinalAnswer -.-> Log[(SQLite Logging)]
```

## Historical Custom Component (Deprecated)

During early R&D, we attempted to train a tiny 124M parameter GPT-2 model from scratch for Tier 1 to prove extreme cost reduction.

```mermaid
flowchart LR
    HF[HuggingFace GPT-2 Weights] --> CPT[Causal Continuation]
    CPT --> SFT[Supervised Fine-Tuning QA]
    SFT --> LoRA[Q/V LoRA Rank 8]
    LoRA --> Probe[Capability Probe]
    Probe --> Result([Result: Hallucination Heavy])
```

The resulting model was structurally coherent but factually nonsensical on unseen queries. It was replaced by the Qwen-0.5B instruct model for the practical benchmark.
