"""We use a simple word2vec model that gets a document vector by averaging all words in the document.

Since we are getting vectors for scientific documents, we must load a vocabulary to train the model from scratch. Therefore we define different subclasses for each scientific field, which may differ substantially by vocabulary.

There exists a Doc2Vec module by gensim, but it seems that empirically Word2Vec + averaging can do just as well; furthermore, we're mainly interested in a simple baseline to compare with sophisticated embeddings.

Links:
    gensim: https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#
"""

import os
import time

import numpy as np

from .vectorizer import Vectorizer
from .preprocessing import CustomPreprocessor
from tqdm import tqdm
from typing import Callable

from gensim.models import Word2Vec

from multiprocessing import cpu_count


EMBEDDING_DIM = 300

# Training data for vocabulary

current_file_abs_path = os.path.dirname(os.path.abspath(__file__))
corpora_path = os.path.join(current_file_abs_path, "corpora")
ASTROPHYSICS_CORPUS = "astro_small.txt"
DEFAULT_CORPUS = os.path.join(corpora_path, ASTROPHYSICS_CORPUS)


class Word2VecVectorizer(Vectorizer):
    def __init__(
        self,
        corpus_path: str,
        model_path: str = None,
        vector_size: int = EMBEDDING_DIM,
        window: int = 5,
        min_count: int = 2,
        workers: int = cpu_count(),
        epochs: int = 10,
        tokenizer: Callable[[str], list[str]] = None,
        **kwargs,
    ) -> None:
        """Construct a Word2Vec based document embedding model from a corpus."""
        super().__init__()

        if tokenizer is None:
            preprocessor = CustomPreprocessor()
            self.tokenizer = preprocessor.custom_preprocess

        if (model_path is None) or (not os.path.exists(model_path)):
            start = time.time()
            # Assume the file is line-based, and one document per line
            print(
                f"Loading and tokenizing data from {corpus_path} for vocabulary and training..."
            )
            sentences = [
                self.tokenizer(line)
                for line in tqdm(open(corpus_path), desc="tokenizing lines")
            ]

            print(f"Training Word2Vec model...")
            model = Word2Vec(
                sentences=sentences,
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                workers=workers,
                epochs=epochs,
            )
            duration = time.time() - start
            print(f"Loaded corpus and trained model in {duration:.2f} seconds.")
        else:
            print(f"Loading saved Word2Vec model from {model_path}.")
            model = Word2Vec.load(model_path)

        self.model = model

        # We don't plan to train the model any further, so we call `init_sims` to make the model much more memory-efficient
        # If `replace` is set, forget the original vectors and only keep the normalized ones = saves lots of memory!
        self.model.init_sims(replace=True)

        # write model to disk to load later and save time
        if model_path is not None:
            print(f"Saving Word2Vec model at {model_path}.")
            self.model.save(model_path)

    def embed_documents(self, docs: list[str], **kwargs) -> np.ndarray:
        """Embed a list of documents (raw text) into word2vec document vectors by averaging the word vectors in each of the documents.

        Since there's no speedup via batching like there is in pytorch models, we iterate one document at a time.
        """
        means = []
        success_indices = []
        failed_indices = []
        for i, doc in tqdm(
            enumerate(docs),
            desc="embedding documents",
            leave=True,
            total=len(docs),
        ):
            mean = np.mean(
                [
                    self.model.wv[word]
                    for word in self.tokenizer(doc)
                    if word in self.model.wv
                ],  # shape `(300,)`
                axis=0,
            )
            # OOV items become NaN
            if np.isnan(mean).any():
                failed_indices.append(i)
            else:
                means.append(mean)
                success_indices.append(i)

        return {
            "embeddings": np.array(means),
            "success_indices": np.array(success_indices, dtype=int),
            "fail_indices": np.array(failed_indices, dtype=int),
        }
