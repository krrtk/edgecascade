from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
import torch
import traceback
import sys

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Workaround for Windows AppLocker blocking compiled sklearn DLLs inside transformers
try:
    import sklearn.metrics
except Exception:
    import types, importlib.machinery
    s = types.ModuleType('sklearn')
    s.__spec__ = importlib.machinery.ModuleSpec('sklearn', None)
    m = types.ModuleType('sklearn.metrics')
    m.__spec__ = importlib.machinery.ModuleSpec('sklearn.metrics', None)
    m.roc_curve = lambda *a, **k: None
    sys.modules['sklearn'] = s
    sys.modules['sklearn.metrics'] = m

from inference.practical_local_model import PracticalLocalModel
from inference.pipeline import EdgeCascadePipeline
from api.database import init_db, log_request, update_feedback, get_metrics

app = FastAPI(title="EdgeCascade API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None

class QueryRequest(BaseModel):
    question: str
    domain: str

class FeedbackRequest(BaseModel):
    request_id: str
    feedback: str  # "positive" or "negative"

@app.on_event("startup")
async def startup_event():
    global pipeline
    
    # Initialize SQLite DB
    init_db()
    
    print("==================================================")
    print("Starting EdgeCascade API...")
    print("==================================================")
    
    # Check if we should skip loading for fast dev, but by default we load.
    # To save time in constrained environments we can mock it, but the user said:
    # "Use the ACTUAL existing EdgeCascade pipeline."
    try:
        print("[1/3] Loading Practical Local Model (Qwen-0.5B)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        local_model = PracticalLocalModel(device=device)
        
        print("[2/3] Initializing TF-IDF Retriever...")
        rag_paths = {
            "isro": "data/retrieval_final/isro_index.pkl",
            "dpdpa": "data/retrieval_final/dpdpa_index.pkl"
        }
        
        print("[3/3] Initializing Pipeline...")
        pipeline = EdgeCascadePipeline(
            local_model=local_model,
            tokenizer=None,
            device=device,
            rag_paths=rag_paths
        )
        print("Pipeline initialization complete.")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        traceback.print_exc()
        raise e

@app.get("/health")
def health_check():
    return {"status": "ok", "pipeline_ready": pipeline is not None}

@app.post("/query")
def query_endpoint(req: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
    req_id = str(uuid.uuid4())
    
    try:
        stats = pipeline.run(req.question, req.domain, mode="cascade")
        
        # Log to SQLite
        log_request(req_id, req.question, req.domain, stats)
        
        return {
            "request_id": req_id,
            "answer": stats.get("answer", ""),
            "tier": stats.get("tier", "unknown"),
            "rag_used": stats.get("retrieval_used", False),
            "remote_used": stats.get("remote_used", False),
            "latency_ms": int(stats.get("latency_total", 0) * 1000)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
def feedback_endpoint(req: FeedbackRequest):
    success = update_feedback(req.request_id, req.feedback)
    if not success:
        raise HTTPException(status_code=404, detail="Request ID not found")
    return {"status": "ok"}

@app.get("/metrics")
def metrics_endpoint():
    try:
        return get_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files for the frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")
