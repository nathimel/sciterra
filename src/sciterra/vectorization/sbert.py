"""We use the acronym SBERT as a catch-all for BERT-based sentence transformers. In particular, we use a lightweight/fast version of one the top-performing model.

Links:
    sbert: https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models.
    HF: https://huggingface.co/sentence-transformers
"""

import torch
import numpy as np
from .vectorizer import Vectorizer
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

MPS_DEVICE = torch.device("mps")

# MODEL_PATH = "bert-base-nli-mean-tokens" # NOTE: while Liu and Xu (2020) use this model in a metascience context, huggingface recommends a more recent sentence transformer.
MODEL_PATH = "all-MiniLM-L6-v2" # All-round model tuned for many use-cases. Trained on a large and diverse dataset of over 1 billion training pairs. Listed as rank 50 on https://huggingface.co/spaces/mteb/leaderboard on 10/11/2023 with an average of 56; rank 1 achieved 64, bert-base-uncased achieved 34; GPT embedding ada-002 achieved 60.
EMBEDDING_DIM = 384
MAX_SEQ_LENGTH = 256

class SBERTVectorizer(Vectorizer):

    def __init__(self, device="cuda") -> None:
        # Get the model
        self.model = SentenceTransformer(MODEL_PATH)

        # set device to GPU
        if device == "mps":
            self.device = MPS_DEVICE
        elif device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {self.device}.")
        self.model.to(self.device)

        # Put the model in "evaluation" mode
        self.model.eval()
        super().__init__()

    def embed_documents(self, docs: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed a list of documents (raw text) into SBERT vectors, by batching.

        Args:
            docs: the documents to embed.

        Returns:
            a numpy array of shape `(num_documents, 384)`
        """

        embeddings = []

        pbar = tqdm(
            total=len(docs),
            desc="embedding documents",
            leave=True,
        )

        for i in range(0, len(docs), batch_size):
            batch = docs[i : min(len(docs), i + batch_size)]

            # no need to convert anything or dig inside model for outputs
            batched_embeddings = self.model.encode(batch)

            # Collect batched embeddings
            embeddings.extend(batched_embeddings)

            pbar.update(batch_size)
        pbar.close()

        return np.array(embeddings)
