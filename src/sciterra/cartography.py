"""Functions for manipulating an atlas based on the document embeddings of the abstracts of its publications."""

import numpy as np

from typing import Type

from atlas import Atlas
from librarians.librarian import Librarian
from vectorization.vectorizer import Vectorizer


def atlas_to_embeddings(atl: Atlas, vectorizer: Vectorizer) -> np.ndarray:
    """Obtain document embeddings for all publications in an atlas."""
    return vectorizer.embed_documents([pub.abstract for pub in atl.publications])

def expand(atl: Atlas) -> Atlas:
    """Expand an atlas by retrieving a list of publications resulting from traversal of the citation network."""
    pass
