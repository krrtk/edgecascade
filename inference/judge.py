import re
import math
import numpy as np
import os
import json
import time
import requests
from collections import defaultdict

def load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line:
                    k, v = line.strip().split('=', 1)
                    os.environ[k] = v

load_env()

class APIJudge:
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key not found")
        if os.environ.get("GROQ_API_KEY"):
            self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
            self.model = "qwen/qwen3.8-27b"
        else:
            self.endpoint = "https://api.openai.com/v1/chat/completions"
            self.model = "gpt-4o-mini"
            
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def judge(self, question, answer, context=None, **kwargs):
        ans_start = answer.find("Answer:")
        if ans_start != -1:
            ans_text = answer[ans_start + 7:].strip()
        else:
            ans_text = answer.strip()
            
        system_prompt = (
            "You are a strict evaluation judge routing answers in a semantic pipeline. "
            "Your job is to determine if a generated answer is sufficiently good to PASS. "
            "An answer should PASS only if it is coherent, directly relevant to the question, "
            "and does not contain unsupported, circular, repetitive, or fabricated claims.\n\n"
        )
        
        if context:
            system_prompt += (
                "Treat the retrieved context as the sole evidence source. If important factual "
                "claims in the answer are unsupported by the provided context, FAIL the answer.\n\n"
            )
            
        system_prompt += (
            "You MUST output exactly valid JSON with the following structure:\n"
            "{\n"
            "  \"pass\": true/false,\n"
            "  \"score\": int (1-5),\n"
            "  \"reason\": \"Brief explanation\"\n"
            "}\n"
        )
        
        user_prompt = f"Question: {question}\nGenerated Answer: {ans_text}\n"
        if context:
            user_prompt += f"Context: {context}\n"
            
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }

        for attempt in range(3):
            try:
                response = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=10)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                result = json.loads(content)
                return {
                    "pass": bool(result.get("pass", False)),
                    "score": int(result.get("score", 1)),
                    "reason": str(result.get("reason", "Unknown reason"))
                }
            except Exception as e:
                if attempt == 2:
                    return {"pass": False, "score": 1, "reason": f"API Error: {str(e)}"}
                time.sleep(1)


class SemanticRoutingJudge:
    def __init__(self):
        self.confidence_threshold = -1.2 
        
    def judge(self, question, answer, context=None, log_probs=None, **kwargs):
        # Kept for compatibility but we are using EvidenceGate now
        return {"pass": True, "score": 4, "reason": "Bypassed by EvidenceGate"}

def get_judge():
    if os.environ.get("ROUTING_JUDGE") == "api" and (os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        return APIJudge()
    return SemanticRoutingJudge()

class LLMEvaluationJudge:
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key not found")
        if os.environ.get("GROQ_API_KEY"):
            self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
            self.model = "qwen/qwen3.8-27b"
            self.provider = "Groq"
        else:
            self.endpoint = "https://api.openai.com/v1/chat/completions"
            self.model = "gpt-4o-mini"
            self.provider = "OpenAI"
            
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.call_count = 0

    def evaluate_batch(self, question, ground_truth, candidates):
        """
        candidates: dict mapping system_name to answer string
        Returns dict mapping system_name to result dict.
        """
        self.call_count += 1
        system_prompt = (
            "You are an expert evaluation judge determining if candidate answers are semantically correct given a reference ground-truth answer. "
            "Evaluate SEMANTIC correctness, not exact string matching. "
            "Distinguish between: correct (1.0), partially correct (0.5), incorrect (0.0). "
            "Output EXACTLY valid JSON with the following structure, keyed by the system name:\n"
            "{\n"
            "  \"local\": {\"correctness\": \"correct/partially correct/incorrect\", \"score\": 1.0, \"reason\": \"...\"},\n"
            "  \"rag\": {\"correctness\": \"correct/partially correct/incorrect\", \"score\": 1.0, \"reason\": \"...\"},\n"
            "  \"remote\": {\"correctness\": \"correct/partially correct/incorrect\", \"score\": 1.0, \"reason\": \"...\"},\n"
            "  \"cascade\": {\"correctness\": \"correct/partially correct/incorrect\", \"score\": 1.0, \"reason\": \"...\"}\n"
            "}\n"
        )
        
        user_prompt = f"Question: {question}\nReference Answer: {ground_truth}\n"
        for sys_name, ans in candidates.items():
            user_prompt += f"Candidate Answer ({sys_name}): {ans}\n"
            
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }

        # 1 retry only for faster failure
        max_retries = 1
        base_delay = 2
        for attempt in range(max_retries):
            try:
                response = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=20)
                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.exceptions.RequestException(f"API Error {response.status_code}")
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                result = json.loads(content)
                
                final_res = {}
                for sys_name in candidates.keys():
                    sys_res = result.get(sys_name, {})
                    final_res[sys_name] = {
                        "status": str(sys_res.get("correctness", "incorrect")).lower(),
                        "score": float(sys_res.get("score", 0.0)),
                        "reason": str(sys_res.get("reason", "Unknown reason")),
                        "correct": str(sys_res.get("correctness", "incorrect")).lower() == "correct"
                    }
                return final_res
            except Exception as e:
                if attempt == max_retries - 1:
                    return {
                        sys_name: {"status": "API_ERROR", "score": 0.0, "reason": f"API Error: {str(e)}", "correct": False}
                        for sys_name in candidates.keys()
                    }
                time.sleep(base_delay * (2 ** attempt))

class RemoteLLM:
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            print("Warning: No API key found for RemoteLLM. Tier-3 will fail.")
        
        if os.environ.get("GROQ_API_KEY"):
            self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
            self.model = "qwen/qwen3.8-27b"
        else:
            self.endpoint = "https://api.openai.com/v1/chat/completions"
            self.model = "gpt-4o-mini"
            
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    def generate(self, question, ground_truth=None, context=None):
        if not self.api_key:
            return "[REMOTE_ERROR] No API key available."
            
        system_prompt = "You are a helpful and accurate assistant. Please answer the user's question concisely."
        user_prompt = f"Question: {question}\n"
        if context:
            user_prompt += f"Context: {context}\n"
            
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 150
        }

        # 1 retry only for faster failure
        max_retries = 1
        base_delay = 2
        for attempt in range(max_retries):
            try:
                response = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=20)
                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.exceptions.RequestException(f"API Error {response.status_code}")
                response.raise_for_status()
                data = response.json()
                answer = data["choices"][0]["message"]["content"].strip()
                return f"[REMOTE] {answer}"
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"[REMOTE_ERROR] API Error: {str(e)}"
                time.sleep(base_delay * (2 ** attempt))
