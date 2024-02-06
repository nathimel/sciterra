"""GPT-2 is a large causal language model that achieved SOTA in many NLP tasks before its successors created by OpenAI.

Links:
    - Paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    - HF: https://huggingface.co/docs/transformers/en/model_doc/gpt2
"""


# See https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Model

import torch
import numpy as np
from .vectorizer import Vectorizer
from tqdm import tqdm

from transformers import AutoTokenizer, GPT2Model, GPT2TokenizerFast


MPS_DEVICE = torch.device("mps")

# This is the hidden dimension size
EMBEDDING_DIM = 768

# Default is small, otherwise memory limits become a problem
BATCH_SIZE = 8


class GPT2Vectorizer(Vectorizer):
    def __init__(self, device="cuda", **kwargs) -> None:
        # Get tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        # Need to specify padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get model
        self.model = GPT2Model.from_pretrained("gpt2")

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

    def embed_documents(
        self,
        docs: list[str],
        batch_size: int = BATCH_SIZE,
    ) -> dict[str, np.ndarray]:
        """Embed a list of documents (raw text) into GPT-2 vectors, by batching.

        Args:
            docs: the documents to embed.

        Returns:
            a numpy array of shape `(num_documents, embedding_dim)`

        """
        if batch_size is None:
            batch_size = BATCH_SIZE

        embeddings = []

        pbar = tqdm(
            total=len(docs),
            desc="embedding documents",
            leave=True,
        )

        for i in range(0, len(docs), batch_size):
            batch = docs[i : min(len(docs), i + batch_size)]

            # A note on padding from Thomas Wolf: https://github.com/huggingface/transformers/issues/808#issuecomment-522932583

            # Tokenize the batch
            encoded = self.tokenizer(
                batch,
                add_special_tokens=True,
                padding=True,  # pad up to length of longest abstract
                truncation=True,  # max length 1024 tokens
                return_tensors="pt",
            )
            # each encoded item of shape [64, 1024]
            input_ids = encoded["input_ids"]
            assert input_ids.size()[-1] <= 1024

            # Put data on GPU
            for k, v in encoded.items():
                encoded[k] = v.to(self.device)

            # Run the text through SciBERT,
            # collecting all of the hidden states produced from all 12 layers.
            with torch.no_grad():
                outputs = self.model(  # discard logits
                    **encoded,
                )

            # Get the embeddings of each final token in the batch

            # Get the full last hidden state,
            # shape [batch_size, sequence_length, hidden_size=768]
            last_hidden_state = outputs.last_hidden_state

            # Get the varying sequence lengths,
            # shape [batch_size,]
            sequence_lengths = torch.tensor(
                [
                    torch.nonzero(token_ids.eq(self.tokenizer.pad_token_id))[0].item()
                    + 1
                    if token_ids.eq(self.tokenizer.pad_token_id).any()
                    else len(token_ids)
                    for token_ids in input_ids
                ]
            )

            # Get embeddings of each final token,
            # shape [batch_size, hidden_size]
            last_hidden_states = torch.stack(
                [
                    last_hidden_state[i, sequence_lengths[i] - 1, :]
                    for i in range(len(sequence_lengths))
                ]
            )

            # Move to the CPU and convert to numpy ndarray
            batched_embeddings = last_hidden_states.detach().cpu().numpy()

            # Collect batched embeddings
            embeddings.extend(batched_embeddings)

            pbar.update(batch_size)
        pbar.close()

        # We don't deal with OOV, so we always return full list of ids
        return {
            "embeddings": np.array(embeddings),
            "success_indices": np.arange(len(embeddings), dtype=int),
            "fail_indices": np.array([], dtype=int),
        }
