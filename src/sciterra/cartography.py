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
            warnings.warn(f"Some abstracts were not available. Found {len(valid_pubs)} nonempty abstracts out of {len(atl)} total publications.")

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

    def expand(
        self, 
        atl: Atlas,
        center: str = None,
        n_pubs_max: int = 4000, 
        n_sources_max: int = None, 
        ) -> Atlas:
        """Expand an atlas by retrieving a list of publications resulting from traversal of the citation network.
        

        Args: 
            atl: the atlas containing the region to expand

            center: (if given) center the search on this publication, preferentially searching related publications.

            n_pubs_max: maximum number of publications allowed in the expansion.

            n_sources_max: maximum number of publications (already in the atlas) to draw references and citations from.

        Returns:
            atl_expanded: the expanded atlas
        """
        if center is None:
            expand_keys = atl.publications.keys()
        else:
            raise NotImplementedError("Expanding around a center not implemented yet.")
        
        if n_sources_max is not None:
            expand_keys = expand_keys[:n_sources_max]

        # Get identifiers for the expansion
        # For each publication corresponding to an id in `expand_keys`, collect the ids corresponding to the publication's references and citations.
        existing_keys = set(atl.publications.keys())
        ids = set()
        for key in expand_keys:
            # NOTE: the below line may need to be broken into a larger function to handle edge cases
            ids_i = set(atl[key].references + atl[key].citations)
            # Prune for obvious overlap
            ids += ids_i - existing_keys
            # Break when the search is centered and we're maxed out
            if len( ids ) > n_pubs_max and center is not None:
                break
        
        if not ids:
            print("Overly-restrictive search, no ids to retrive.")

        # Sample to account for max number of publications we want to retrieve
        if len( ids ) > n_pubs_max:
            ids = np.random.choice(ids, n_pubs_max, replace=False)

        print(f"Expansion will include {len(ids)} new publications.")

        # Retrieve publications from ids using a librarian
        new_publications = self.librarian.query_publications(ids)
        
        # New atlas
        atl_exp = Atlas(new_publications)

        # Update the new atlas
        atl_exp.publications.update(atl.publications)

        return atl_exp
