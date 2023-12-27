"""Bag of words document embedder. Unlike cc vectorization, we fix the dimension of the embeddings to be the same; this requires us to fix the vocabulary, so for consistency we do so via the same method as the Word2Vec vocabulary construction."""


import os
import time

import numpy as np
from numpy import ndarray

from .vectorizer import Vectorizer
from .word2vec import Word2VecVectorizer
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer


from multiprocessing import cpu_count


# Training data for vocabulary

current_file_abs_path = os.path.dirname(os.path.abspath(__file__))
corpora_path = os.path.join(current_file_abs_path, "corpora")
ASTROPHYSICS_CORPUS = "astro_small.txt"
DEFAULT_CORPUS = os.path.join(corpora_path, ASTROPHYSICS_CORPUS)


class BOWVectorizer(Vectorizer):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Construct a bag-of-words document vectorizer."""
        # this part is slow because involves training the word2vec model
        self.word2vec_vectorizer = Word2VecVectorizer(*args, **kwargs)
        self.vocabulary = self.word2vec_vectorizer.model.wv.key_to_index
        self.embedding_dim = len(self.vocabulary)
        self.count_vectorizer = CountVectorizer(vocabulary=self.vocabulary)

    def embed_documents(self, docs: list[str]) -> dict[str, ndarray]:
        """Embed a list of documents (raw text) into bow document vectors using scikit-learn's CountVectorizer.

        Args:
            docs: the documents to embed.

        Returns:
            a numpy array of shape `(num_documents, len(self.vocabulary))`
        """
        # We could process all documents at once, but we sacrifice some speed to see a progress bar.
        embeddings = np.array(
            [
                self.count_vectorizer.transform([doc]).toarray().sum(axis=0)
                for doc in tqdm(docs, desc="embedding documents", leave=True)
            ]
        )

        # Find the indices of documents that had at least 1 in-vocab token
        success_indices = np.where(np.sum(embeddings, axis=1) > 0)[0]
        fail_indices = np.setdiff1d(np.arange(len(docs)), success_indices)

        return {
            "embeddings": embeddings[success_indices],
            "success_indices": success_indices,
            "fail_indices": fail_indices,
        }
