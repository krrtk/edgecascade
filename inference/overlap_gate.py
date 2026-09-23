import re

class RetrievalOverlapGate:
    def __init__(self, threshold=0.10):
        self.threshold = threshold

    def evaluate(self, retrieval_score):
        """
        Takes the similarity score from the retriever.
        If score > threshold, it's meaningful.
        """
        return retrieval_score >= self.threshold

class EvidenceGate:
    def __init__(self, retrieval_threshold=0.10, overlap_threshold=0.2):
        self.retrieval_threshold = retrieval_threshold
        self.overlap_threshold = overlap_threshold

    def evaluate(self, query, answer, context, retrieval_score):
        """
        Evaluates if the answer is supported by the context.
        Returns (PASS (bool), Reason (str))
        """
        if retrieval_score < self.retrieval_threshold:
            return False, f"Retrieval score {retrieval_score:.2f} below threshold {self.retrieval_threshold}"
            
        if not context:
            return False, "No context provided"

        # Check for numbers in answer not in context
        ans_nums = re.findall(r'\b\d+(?:\.\d+)?\b', answer)
        for num in ans_nums:
            if num not in context:
                return False, f"Unsupported number '{num}'"
                
        # Check for capitalized entities in answer not in context
        ans_entities = re.findall(r'\b[A-Z][a-z]{3,}\b', answer)
        common = {"The", "This", "That", "It", "They", "What", "How", "When", "Where", "Why", "Yes", "No"}
        for ent in ans_entities:
            if ent not in common and ent.lower() not in context.lower():
                return False, f"Unsupported entity '{ent}'"
                
        # Word overlap
        words1 = set(re.findall(r'\b\w+\b', context.lower()))
        words2 = set(re.findall(r'\b\w+\b', answer.lower()))
        if not words2: 
            return False, "Empty answer"
        
        stopwords = {"the", "is", "a", "of", "in", "and", "to", "for", "with", "on", "at", "by", "from", "it", "has", "width"}
        ans_content = words2 - stopwords
        
        if not ans_content: 
            # If the answer only contains stopwords, we can't accept it as highly substantive
            return False, "Answer contains no substantive words"
        
        overlap = len(ans_content.intersection(words1))
        overlap_ratio = overlap / len(ans_content)
        
        if overlap_ratio < self.overlap_threshold:
            return False, f"Low semantic overlap ({overlap_ratio:.2f} < {self.overlap_threshold})"
            
        return True, "Evidence strongly supports answer"
