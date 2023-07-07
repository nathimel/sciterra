"""Functions for manipulating an atlas based on the document embeddings of the abstracts of its publications."""

import bibtexparser
import warnings

import numpy as np

from typing import Type, Any

from .atlas import Atlas
from .librarians.librarian import Librarian
from .publication import Publication
from .vectorization.vectorizer import Vectorizer
from .vectorization.projection import Projection

from sklearn.metrics.pairwise import cosine_similarity


class Cartographer:
    """A basic wrapper for obtaining and updating atlas projections."""

    def __init__(
        self, 
        librarian: Librarian = None,
        vectorizer: Vectorizer = None,
        ) -> None:

        self.librarian = librarian
        self.vectorizer = vectorizer

    def bibtex_to_atlas(self, bibtex_fp: str, *args, **kwargs) -> Atlas:
        """Convert a bibtex file to an atlas, by parsing each entry for an identifier, and querying an API for publications using `self.librarian`.

        NOTE: the identifiers in the corresponding atlas will be API-specific ids; there is no relationship between the parsed id used to query papers (e.g. 'DOI:XYZ' in the case of SemanticScholar) and the resulting identifier associated with the resulting Publication object, (a paperId, a bibcode, etc.) Therefore, the purpose of using the `bibtex_to_atlas` method is primarily for initializing literature exploration in a human-readable way. If you want to obtain as many publications as identifiers supplied, you need to use `get_publications`.
        
        Args:
            bibtex_fp: the filepath where the bibtex file is saved.

            args and kwargs are passed to `get_publications`.
        """
        with open(bibtex_fp, "r") as f:
            bib_database = bibtexparser.load(f)

        # Retrieve the identifier from each bibtex entry
        identifiers = [self.librarian.bibtex_entry_identifier(entry) for entry in bib_database.entries]
        identifiers = [id for id in identifiers if id is not None]
        if len(identifiers) < len(bib_database.entries):
            warnings.warn(f"Only obtained {len(identifiers)} publications out of {len(bib_database.entries)} total due to missing identifiers.")

        # Query
        results = self.librarian.get_publications(identifiers, *args, **kwargs)
        # Validate
        publications = [
            result for result in results
            if (
            result is not None
            and result.publication_date is not None
            and result.abstract is not None
            # identifier will never be none            
            )
        ]
        if len(publications) < len(identifiers):
            warnings.warn(f"Only obtained {len(publications)} publications out of {len(identifiers)} total due to querying-related errors or missing abstracts.")

        # Construct atlas
        atl = Atlas(publications)
        return atl

    def project(self, atl: Atlas) -> Atlas:
        """Update an atlas with its projection, i.e. the document embeddings for all publications, removing publications with no abstracts.
        
        Args:
            atl: the Atlas containing publications to project to document embeddings

        Returns:
            the updated atlas containing all nonempty-abstract-containing publications and their projection
        """
        projection = self.get_projection(atl.publications)
        publications = {k:v for k,v in atl.publications.items() if k in projection.identifier_to_index}

        invalid = set(atl.publications.keys()) - set(publications.keys())
        if invalid:
            warnings.warn(f"Removing {len(invalid)} publications from atlas after projection.")
            breakpoint() # this is supposed to prevent the index error but it's not
            # why is resetting the number of publications not working?
        breakpoint()
        atl.publications = publications
        atl.projection = projection
        return atl
    
    def get_projection(self, publications: dict[str, Publication]) -> Projection:
        """Obtain document embeddings for all publications that have abstracts using `self.vectorizer`.
        
        Args:
            atl: the Atlas containing publications to project to document embeddings

        Returns:
            a Projection of all publications with non-empty abstracts
        """
        id_to_idx = {}
        idx_to_id = []
        valid_pubs = []
        invalid = 0
        for idx, (id, pub) in enumerate(publications.items()):
            if pub.abstract is None:
                invalid += 1
                continue
            id_to_idx[str(pub)] = idx
            idx_to_id.append(str(pub))
            valid_pubs.append(pub)
        
        if invalid:
            warnings.warn(f"Some abstracts were not available. Found {len(valid_pubs)} nonempty abstracts out of {len(publications)} total publications.")

        embeddings = None
        if valid_pubs:
            embeddings = self.vectorizer.embed_documents([pub.abstract for pub in valid_pubs])
        
        if embeddings is None:
            warnings.warn(f"Obtained no publication embeddings.")
        
        return Projection(
            id_to_idx,
            tuple(idx_to_id),
            embeddings,
        )

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
        existing_keys = set(atl.publications.keys())
        if center is None:
            expand_keys = existing_keys
        else:
            if atl.projection is None:
                atl = self.project(atl)

            # cosine similarity matrix
            # breakpoint()
            # for some even though projection should have only 148 items, the identifier_to_index map contains indices greater than this. They should just describe the embeddings, but maybe I erroroneously defined them before i defined embeddings.
            cospsi_matrix = cosine_similarity(
                atl.projection.identifiers_to_embeddings([center]),
                atl.projection.embeddings,
            )
            sort_inds = np.argsort(cospsi_matrix)[1:] # exclude the center
            expand_keys = atl.projection.indices_to_identifiers(sort_inds)

            if len(expand_keys) < len(existing_keys):
                expand_keys = set(expand_keys).union(existing_keys)
        
        if n_sources_max is not None:
            expand_keys = expand_keys[:n_sources_max]

        # Get identifiers for the expansion
        # For each publication corresponding to an id in `expand_keys`, collect the ids corresponding to the publication's references and citations.
        ids = set()
        for key in expand_keys:
            ids_i = set(atl[key].references + atl[key].citations)
            # Prune for obvious overlap
            ids.update(ids_i - existing_keys)
            # Break when the search is centered and we're maxed out
            if len( ids ) > n_pubs_max and center is not None:
                break
        ids = list(ids)

        if not ids:
            print("Overly-restrictive search, no ids to retrive.")

        # Sample to account for max number of publications we want to retrieve
        if len( ids ) > n_pubs_max:
            ids = np.random.choice(ids, n_pubs_max, replace=False)

        print(f"Expansion will include {len(ids)} new publications.")

        # Retrieve publications from ids using a librarian
        new_publications = self.librarian.get_publications(ids)
        
        # New atlas
        atl_exp = Atlas(new_publications)

        # Update the new atlas
        atl_exp.publications.update(atl.publications)

        return atl_exp
