"""Functions for manipulating an atlas based on the document embeddings of the abstracts of its publications."""

import warnings

import numpy as np

from typing import Type

from .atlas import Atlas
from .librarians.librarian import Librarian
from .vectorization.vectorizer import Vectorizer

class Cartographer:
    """A basic wrapper for obtaining and updating atlas projections."""

    def __init__(
        self, 
        librarian: Librarian,
        vectorizer: Vectorizer,
        ) -> None:

        self.librarian = librarian
        self.vectorizer = vectorizer

    def project(self, atl: Atlas) -> np.ndarray:
        """Obtain document embeddings for all publications in an atlas that have abstracts.
        
        Args:
            atl: the Atlas containing publications to project to document embeddings

        Returns:
            a dict of the form 
                {
                    "identifiers_to_indices": a dict[str, int] mapping a publication identifier to its embedding index,
                    "indices_to_identifiers": a tuple mapping an embedding index to its str publication identifier
                    "embeddings": a np.ndarray of shape `(num_publications, embedding_dim)`
                }
        """
        id_to_idx = {}
        idx_to_id = []
        valid_pubs = []
        invalid = 0
        for idx, pub in enumerate(atl.publications):
            if pub.abstract is None:
                invalid += 1
                continue
            id_to_idx[str(pub)] = idx
            idx_to_id.append(str(pub))
            valid_pubs.append(pub)
        
        if invalid:
            warnings.warn(f"Found {len(valid_pubs)} nonempty abstracts out of {len(atl)} total publications.")

        embeddings = None
        if valid_pubs:
            embeddings = self.vectorizer.embed_documents([pub.abstract for pub in valid_pubs])
        
        if embeddings is None:
            warnings.warn(f"Obtained no publication embeddings.")

        return {
            "identifiers_to_indices": id_to_idx,
            "indices_to_identifiers": tuple(idx_to_id),
            "embeddings": embeddings,
        }

    def expand(atl: Atlas) -> Atlas:
        """Expand an atlas by retrieving a list of publications resulting from traversal of the citation network."""
        pass
