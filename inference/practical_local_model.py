import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import math
import os
import time
import math

class PracticalLocalModel:
    """
    Wrapper for an open-weight instruction-tuned Hugging Face model
    to be used as the practical Tier-1/Tier-2 local model in EdgeCascade.
    """
    def __init__(self, model_name=None, device="cpu"):
        self.device = device
        if model_name is None:
            model_name = os.environ.get("PRACTICAL_MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
        self.model_name = model_name
        print(f"Loading {self.model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
        self.model.eval()
        
    def generate(self, prompt, max_new_tokens=100, temperature=0.7, top_k=40, return_probs=True):
        """
        Generates text and returns (response, log_probs) if return_probs=True
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                do_sample=True if temperature > 0 else False,
                return_dict_in_generate=True,
                output_scores=return_probs,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        log_probs_list = []
        if return_probs and outputs.scores:
            # Approximate log_probs for compatibility with judge
            # For each generated token, find its log prob
            for i, score in enumerate(outputs.scores):
                probs = torch.softmax(score[0], dim=-1)
                token_id = generated_ids[i]
                token_prob = probs[token_id].item()
                log_probs_list.append(math.log(max(token_prob, 1e-10)))
                
        if return_probs:
            return response, log_probs_list
        return response
