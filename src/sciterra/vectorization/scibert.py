"""SciBERT is a BERT model trained on scientific text.

Links:
    Paper: https://aclanthology.org/D19-1371/
    Github:  https://github.com/allenai/scibert
    HF: https://huggingface.co/allenai/scibert_scivocab_uncased
"""

import numpy as np
from vectorizer import Vectorizer
from transformers import AutoTokenizer, AutoModel

class SciBERTVectorizer(Vectorizer):

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        super().__init__()
    
    def embed(self, docs: list[str]) -> np.ndarray:
        """Embed a list of documents into SciBERT vectors by retrieving the 
        
        Args:
            docs: the documents to embed.

        Returns:
            a numpy array of shape `(num_documents, 768)`
        """
        pass