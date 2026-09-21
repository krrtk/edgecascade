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
        # Defaulting to Groq compatible endpoint for speed/cost if available, else OpenAI
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
            "IMPORTANT: Do not reward fluency alone. A confident or fluent answer that contains "
            "unsupported, circular, repetitive, or fabricated claims should FAIL.\n\n"
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
            "Score guide:\n"
            "1 = clearly bad/hallucinated\n"
            "2 = weak / substantially incomplete / contradicts context\n"
            "3 = borderline\n"
            "4 = good / supported\n"
            "5 = strong / highly accurate\n"
            "A PASS generally requires a score of 4 or 5."
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
                # Ensure structure
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
        # Threshold for Tier 1 confidence (avg log-prob)
        self.confidence_threshold = -1.2 
        
    def is_factually_supported(self, context, answer):
        # A lightweight local semantic judge for entailment without large models
        ans_nums = re.findall(r'\b\d+(?:\.\d+)?\b', answer)
        for num in ans_nums:
            if num not in context:
                return False, f"Unsupported number '{num}'"
                
        ans_entities = re.findall(r'\b[A-Z][a-z]{3,}\b', answer)
        common = {"The", "This", "That", "It", "They", "What", "How", "When", "Where", "Why"}
        for ent in ans_entities:
            if ent not in common and ent.lower() not in context.lower():
                return False, f"Unsupported entity '{ent}'"
                
        words1 = set(re.findall(r'\b\w+\b', context.lower()))
        words2 = set(re.findall(r'\b\w+\b', answer.lower()))
        if not words2: return False, "Empty answer"
        
        stopwords = {"the", "is", "a", "of", "in", "and", "to", "for", "with", "on", "at", "by", "from", "it", "has", "width"}
        ans_content = words2 - stopwords
        
        if not ans_content: return True, ""
        
        overlap = len(ans_content.intersection(words1))
        if overlap / len(ans_content) < 0.2:
            return False, "Low semantic overlap with context"
            
        return True, ""

    def judge(self, question, answer, context=None, log_probs=None, **kwargs):
        ans_start = answer.find("Answer:")
        if ans_start != -1:
            ans_text = answer[ans_start + 7:].strip()
        else:
            ans_text = answer.strip()
            
        words = ans_text.lower().split()
        if len(words) < 3:
            return {"pass": False, "score": 1, "reason": "Answer too short or incomplete"}
            
        if len(words) > 10:
            vocab = set(words)
            if len(vocab) / len(words) < 0.3:
                return {"pass": False, "score": 1, "reason": "Highly repetitive generation"}
                
        q_clean = re.sub(r'[^\w\s]', '', question.lower())
        a_clean = re.sub(r'[^\w\s]', '', ans_text.lower())
        if a_clean.startswith(q_clean):
            if len(a_clean) < len(q_clean) + 20: 
                return {"pass": False, "score": 2, "reason": "Answer just repeats the question"}
                
        if context is None:
            if log_probs is not None and len(log_probs) > 0:
                avg_log_prob = sum(log_probs) / len(log_probs)
                if avg_log_prob < self.confidence_threshold:
                    return {"pass": False, "score": 2, "reason": f"Low confidence generation (avg log-prob: {avg_log_prob:.2f})"}
            return {"pass": True, "score": 4, "reason": "Answer is coherent and confident"}
        else:
            supported, reason = self.is_factually_supported(context, ans_text)
            if not supported:
                return {"pass": False, "score": 2, "reason": f"Answer not supported by context: {reason}"}
            return {"pass": True, "score": 4, "reason": "Answer is semantically supported by context"}

def get_judge():
    if os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        return APIJudge()
    print("API judge unavailable. Using heuristic fallback.")
    return SemanticRoutingJudge()

# Mock Remote LLM
class RemoteLLM:
    def __init__(self):
        pass
        
    def generate(self, question, ground_truth):
        """
        Simulates a Remote LLM by returning a perfect answer based on the ground truth.
        """
        return f"[REMOTE] {ground_truth}"
