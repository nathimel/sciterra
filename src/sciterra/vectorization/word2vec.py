"""We use a simple word2vec model that gets a document vector by averaging all words in the document.

Since we are getting vectors for scientific documents, we must load a vocabulary to train the model from scratch. Therefore we define different subclasses for each scientific field, which may differ substantially by vocabulary.

There exists a Doc2Vec module by gensim, but it seems that empirically Word2Vec + averaging can do just as well; furthermore, we're mainly interested in a simple baseline to compare with sophisticated embeddings.

Links:
    gensim: https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#
"""

import numpy as np
import nltk

nltk.download("punkt")

from .vectorizer import Vectorizer
from tqdm import tqdm

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from nltk.tokenize import word_tokenize

from multiprocessing import cpu_count


EMBEDDING_DIM = 300


class Word2VecVectorizer(Vectorizer):
    def __init__(
        self,
        corpus_path: str,
        vector_size: int = EMBEDDING_DIM,
        window: int = 5,
        min_count: int = 2,
        workers: int = cpu_count(),
        epochs: int = 20,
    ) -> None:
        """Construct a Word2Vec based document embedding model from a corpus."""
        super().__init__()

        self.tokenizer = word_tokenize

        # Assume the file is line-based
        print(f"Loading and tokenizing training data from {corpus_path}...")
        sentences = [self.tokenizer(line) for line in open(corpus_path)]

        print(f"Training Word2Vec model...")
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=epochs,
        )

    def embed_documents(self, docs: list[str], **kwargs) -> np.ndarray:
        """Embed a list of documents (raw text) into word2vec document vectors by averaging the word vectors in each of the documents.

        Since there's no speedup via batching like there is in pytorch models, we iterate one document at a time.

        Args:
            docs: the documents to embed.

        Returns:
            a numpy array of shape `(num_documents, 300)`
        """

        return np.array(
            [
                np.mean(
                    [
                        self.model.wv[word]
                        for word in self.tokenizer(doc)
                        if word in self.model.wv
                    ],  # shape `(300,)`
                    axis=0,
                )
                for doc in tqdm(
                    docs,
                    desc="embedding documents",
                    leave=True,
                )
            ]
        )
