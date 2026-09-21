import time
from .retriever import SimpleTFIDFRetriever
from .overlap_gate import RetrievalOverlapGate
from .judge import get_judge, RemoteLLM
from inference.generator import generate_text
from inference.practical_local_model import PracticalLocalModel
class EdgeCascadePipeline:
    def __init__(self, local_model, tokenizer, device, rag_paths):
        self.local_model = local_model
        self.tokenizer = tokenizer
        self.device = device
        self.retriever = SimpleTFIDFRetriever(rag_paths)
        self.gate = RetrievalOverlapGate(threshold=0.10)
        self.judge = get_judge()
        self.remote = RemoteLLM()
        
    def run(self, query, domain, ground_truth=None, mode="cascade"):
        """
        modes: 'local', 'rag', 'remote', 'cascade'
        """
        stats = {
            "tier": "unknown",
            "answer": "",
            "judge_result": "N/A",
            "judge_score": 0,
            "judge_reason": "N/A",
            "retrieval_used": False,
            "remote_used": False,
            "latency_t1": 0.0,
            "latency_t2": 0.0,
            "latency_t3": 0.0,
            "latency_total": 0.0,
            "retrieved_chunks": [],
            "escalation_reason": "N/A",
            "local_answer": "",
            "local_judge_result": "N/A",
            "tier2_answer": "",
            "tier2_judge_result": "N/A"
        }
        
        start_total = time.time()
        
        if mode == "remote":
            stats["remote_used"] = True
            stats["tier"] = "tier3"
            t_start = time.time()
            ans = self.remote.generate(query, ground_truth)
            stats["latency_t3"] = time.time() - t_start
            stats["answer"] = ans
            stats["latency_total"] = time.time() - start_total
            return stats
            
        # Tier 1
        t_start = time.time()
        if isinstance(self.local_model, PracticalLocalModel):
            prompt1 = f"You are answering a question about {domain}. Please be concise and accurate.\n\n### Question:\n{query}\n\n### Answer:\n"
            ans1, log_probs1 = self.local_model.generate(prompt1, max_new_tokens=40, temperature=0.8, return_probs=True)
        else:
            prompt1 = f"### Question:\n{query}\n\n### Answer:\n"
            ans1, log_probs1 = generate_text(self.local_model, self.tokenizer, prompt1, max_new_tokens=40, device=self.device, return_probs=True, temperature=0.8, top_k=40)
            ans1 = ans1.strip()
            if ans1.startswith(prompt1):
                ans1 = ans1[len(prompt1):].strip()
                
        stats["latency_t1"] = time.time() - t_start
        stats["local_answer"] = ans1
        
        if mode == "local":
            stats["tier"] = "tier1"
            stats["answer"] = ans1
            stats["latency_total"] = time.time() - start_total
            return stats
            
        if mode == "cascade":
            j_dict = self.judge.judge(question=query, answer=ans1, log_probs=log_probs1)
            stats["judge_result"] = "PASS" if j_dict.get("pass") else "FAIL"
            stats["local_judge_result"] = stats["judge_result"]
            stats["judge_score"] = j_dict.get("score", 0)
            stats["judge_reason"] = j_dict.get("reason", "")
            
            if j_dict.get("pass"):
                stats["tier"] = "tier1"
                stats["answer"] = ans1
                stats["latency_total"] = time.time() - start_total
                return stats
                
            stats["escalation_reason"] = f"T1 Judge Fail (Score {stats['judge_score']}): {stats['judge_reason']}"
            
        # Retrieval
        ret_results = self.retriever.retrieve(query, top_k=1, domain=domain)
        best_chunk = ret_results[0] if ret_results else None
        
        if mode == "cascade":
            if not best_chunk or not self.gate.evaluate(best_chunk[0]):
                stats["escalation_reason"] += " | Retrieval not meaningful"
                stats["remote_used"] = True
                stats["tier"] = "tier3"
                t_start = time.time()
                ans = self.remote.generate(query, ground_truth)
                stats["latency_t3"] = time.time() - t_start
                stats["answer"] = ans
                stats["latency_total"] = time.time() - start_total
                return stats
                
        # Tier 2 (RAG)
        stats["retrieval_used"] = True
        context = best_chunk[1]["text"] if best_chunk else ""
        stats["retrieved_chunks"] = [best_chunk[1]["chunk_id"]] if best_chunk else []
        
        t_start = time.time()
        if isinstance(self.local_model, PracticalLocalModel):
            prompt2 = f"You are answering a question about {domain} using the provided context.\n\n### Context:\n{context}\n\n### Question:\n{query}\n\n### Answer:\n"
            ans2, log_probs2 = self.local_model.generate(prompt2, max_new_tokens=40, temperature=0.8, return_probs=True)
        else:
            prompt2 = f"### Context:\n{context}\n\n### Question:\n{query}\n\n### Answer:\n"
            ans2, log_probs2 = generate_text(self.local_model, self.tokenizer, prompt2, max_new_tokens=40, device=self.device, return_probs=True, temperature=0.8, top_k=40)
            ans2 = ans2.strip()
            if ans2.startswith(prompt2):
                ans2 = ans2[len(prompt2):].strip()
                
        stats["latency_t2"] = time.time() - t_start
        stats["tier2_answer"] = ans2
        
        if mode == "rag":
            stats["tier"] = "tier2"
            stats["answer"] = ans2
            stats["latency_total"] = time.time() - start_total
            return stats
            
        if mode == "cascade":
            j_dict2 = self.judge.judge(question=query, answer=ans2, log_probs=log_probs2, context=context)
            stats["judge_result"] = "PASS" if j_dict2.get("pass") else "FAIL"
            stats["tier2_judge_result"] = stats["judge_result"]
            stats["judge_score"] = j_dict2.get("score", 0)
            stats["judge_reason"] = j_dict2.get("reason", "")
            
            if j_dict2.get("pass"):
                stats["tier"] = "tier2"
                stats["answer"] = ans2
                stats["latency_total"] = time.time() - start_total
                return stats
                
            stats["escalation_reason"] += f" | T2 Judge Fail (Score {stats['judge_score']}): {stats['judge_reason']}"
            stats["remote_used"] = True
            stats["tier"] = "tier3"
            t_start = time.time()
            ans = self.remote.generate(query, ground_truth)
            stats["latency_t3"] = time.time() - t_start
            stats["answer"] = ans
            stats["latency_total"] = time.time() - start_total
            return stats
