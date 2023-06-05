"""SciBERT is a BERT model trained on scientific text.

Links:
    Paper: https://aclanthology.org/D19-1371/
    Github:  https://github.com/allenai/scibert
    HF: https://huggingface.co/allenai/scibert_scivocab_uncased
"""

import torch
import numpy as np
from .vectorizer import Vectorizer
from tqdm import tqdm

from transformers import BertTokenizerFast, AutoModelForSequenceClassification

MPS_DEVICE = torch.device("mps")

# the SciBERT pretrained model path from Allen AI repo
MODEL_PATH = 'allenai/scibert_scivocab_uncased'

class SciBERTVectorizer(Vectorizer):

    def __init__(self, device = "cuda") -> None:

        # Get tokenizer  
        self.tokenizer = BertTokenizerFast.from_pretrained(
            MODEL_PATH, 
            do_lower_case=True,
        )
        # Get the model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path = MODEL_PATH,
            output_attentions = False,
            output_hidden_states = True,
        )

        # set device to GPU
        if device == "mps":
            self.device = MPS_DEVICE
        elif device == "cuda":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device {self.device}.")
        self.model.to(self.device)

        # Put the model in "evaluation" mode
        self.model.eval()
        super().__init__()
    
    def embed_documents(self, docs: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed a list of documents (raw text) into SciBERT vectors, by batching.
        
        Args:
            docs: the documents to embed.

        Returns:
            a numpy array of shape `(num_documents, 768)`
        """

        embeddings = []
        with tqdm(total = len(docs), desc = "embedding documents",) as pbar:
            for i in range(0, len(docs), batch_size):
                batch = docs[i : min(len(docs), i+batch_size)]

                # Tokenize the batch
                encoded = self.tokenizer(
                    batch,
                    add_special_tokens = True,
                    padding = True, # pad up to length of longest abstract
                    return_tensors = 'pt',
                )

                # Put data on GPU
                for k, v in encoded.items():
                    encoded[k] = v.to(self.device)

                # Run the text through SciBERT, and collect all of the hidden states produced
                # from all 12 layers. 
                with torch.no_grad():        
                    _, encoded_layers = self.model( # discard logits
                        **encoded,
                        return_dict=False,
                    )

                # Extract the embeddings
                # index last (13th) BERT layer before the classifier
                final_hidden_state = encoded_layers[12] # (batch_size, 256, 768)
                # index first token of sequence, [CLS], for our document embeddings
                batched_embeddings = final_hidden_state[:,0,:] # (batch_size, 768)

                # Move to the CPU and convert to numpy ndarray
                batched_embeddings = batched_embeddings.detach().cpu().numpy()

                # Collect batched embeddings
                embeddings.extend(batched_embeddings)
                pbar.update(batch_size)
        
        return np.array(embeddings)
