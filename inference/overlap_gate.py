class RetrievalOverlapGate:
    def __init__(self, threshold=0.10):
        self.threshold = threshold

    def evaluate(self, retrieval_score):
        """
        Takes the similarity score from the retriever.
        If score > threshold, it's meaningful.
        """
        return retrieval_score >= self.threshold
