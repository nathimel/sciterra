"""Base class for vectorizing abstracts."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Vectorizer:

    def __init__(self) -> None:
        pass

    def embed(self, docs: list[str]) -> np.ndarray:
        """Embed a list of documents into document vectors.
        
        Args:
            docs: the documents to embed.

        Returns:
            a numpy array of shape `(num_documents, embedding_dim)`
        """
        raise NotImplementedError
    