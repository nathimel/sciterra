"""Base class for vectorizing abstracts."""

import numpy as np


class Vectorizer:
    def __init__(self) -> None:
        pass

    def embed_documents(self, docs: list[str]) -> dict[str, np.ndarray]:
        """Embed a list of documents into document vectors.

        Args:
            docs: the documents to embed.

        Returns:
            a dict of the form
            {
                "embeddings": a numpy array of shape `(num_successful, embedding_dim)`, containing the document embeddingss

                "success_indices": a numpy array of shape `(num_successful,)`, containing the indices of all the documents for which document embeddings were successfully obtained.

                "fail_indices": a numpy array of shape `(len(docs) - num_successful,)`, containing the indices of all the documents for which document embeddings could not be obtained
            }
            where the indices are with respect to the `docs` list passed.

        """
        raise NotImplementedError
