"""Functions for manipulating an atlas based on the document embeddings of the abstracts of its publications."""

import bibtexparser
import warnings

import numpy as np

from .atlas import Atlas
from ..librarians.librarian import Librarian
from ..vectorization.vectorizer import Vectorizer
from ..vectorization.projection import Projection, merge

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

    ######################################################################
    # Get an Atlas from bibtex
    ######################################################################

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
        identifiers = [
            self.librarian.bibtex_entry_identifier(entry)
            for entry in bib_database.entries
        ]
        identifiers = [id for id in identifiers if id is not None]
        if len(identifiers) < len(bib_database.entries):
            warnings.warn(
                f"Only obtained {len(identifiers)} publications out of {len(bib_database.entries)} total due to missing identifiers."
            )

        # Query
        results = self.librarian.get_publications(identifiers, *args, **kwargs)
        # Validate
        publications = [
            result
            for result in results
            if (
                result is not None
                and result.publication_date is not None
                and result.abstract is not None
                # identifier will never be none
            )
        ]
        if len(publications) < len(identifiers):
            warnings.warn(
                f"Only obtained {len(publications)} publications out of {len(identifiers)} total due to querying-related errors or missing abstracts."
            )

        # Construct atlas
        atl = Atlas(publications)
        return atl

    ######################################################################
    # Project Atlas
    ######################################################################

    def project(self, atl: Atlas) -> Atlas:
        """Update an atlas with its projection, i.e. the document embeddings for all publications using `self.vectorizer`, removing publications with no abstracts.

        Args:
            atl: the Atlas containing publications to project to document embeddings

        Returns:
            the updated atlas containing all nonempty-abstract-containing publications and their projection
        """
        # Only project publications that have abstracts
        atl_filtered = self.filter(atl, attributes=["abstract"])
        invalid = len(atl) - len(atl_filtered)
        if invalid:
            warnings.warn(
                f"Some abstracts were not available. Found {len(atl_filtered)} nonempty abstracts out of {len(atl.publications)} total publications."
            )

        # Project
        embeddings = None
        # get only embeddings for publications not already projected in atlas
        previously_embedded_ids = []
        if atl_filtered.projection is not None:
            previously_embedded_ids = atl_filtered.projection.identifier_to_index
        embed_ids = [
            id for id in atl_filtered.publications if id not in previously_embedded_ids
        ]

        # Embed documents
        if embed_ids:
            embeddings = self.vectorizer.embed_documents(
                [atl_filtered[id].abstract for id in embed_ids]
            )
        if embeddings is None:
            warnings.warn(f"Obtained no new publication embeddings.")

        # create new projection
        projection = Projection(
            identifier_to_index={
                identifier: idx for idx, identifier in enumerate(embed_ids)
            },
            index_to_identifier=tuple(embed_ids),
            embeddings=embeddings,
        )
        # merge existing projection with new projection
        merged_projection = merge(atl_filtered.projection, projection)

        # prepare to overwrite atlas with publications corresponding to updated (merged) projection
        embedded_publications = {
            id: pub
            for id, pub in atl_filtered.publications.items()
            if id in merged_projection.identifier_to_index
        }
        invalid = set(atl_filtered.publications.keys()) - set(
            embedded_publications.keys()
        )
        if invalid:
            warnings.warn(
                f"Removing {len(invalid)} publications from atlas after projection."
            )

        # Overwrite atlas data
        atl_filtered.publications = embedded_publications
        atl_filtered.projection = merged_projection
        return atl_filtered

    ######################################################################
    # Expand Atlas
    ######################################################################

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
        expand_keys = existing_keys
        if center is not None:
            if atl.projection is None:
                atl = self.project(atl)

            if len(atl.projection):
                # cosine similarity matrix
                cospsi_matrix = cosine_similarity(
                    atl.projection.identifiers_to_embeddings([center]),
                    atl.projection.embeddings,
                )
                # get most similar keys from center, including center itself
                sort_inds = np.argsort(cospsi_matrix)[0]  # shape (1, num_pubs)
                expand_keys = atl.projection.indices_to_identifiers(sort_inds)

        if n_sources_max is not None:
            expand_keys = expand_keys[:n_sources_max]

        # Get identifiers for the expansion
        # For each publication corresponding to an id in `expand_keys`, collect the ids corresponding to the publication's references and citations.
        ids = set()
        for key in expand_keys:
            try:
                ids_i = set(atl[key].references + atl[key].citations)
            except ValueError as e:
                breakpoint()
            # Prune for obvious overlap
            ids.update(ids_i - existing_keys)
            # Break when the search is centered and we're maxed out
            if len(ids) > n_pubs_max and center is not None:
                break
        ids = list(ids)

        if not ids:
            print("Overly-restrictive search, no ids to retrive.")

        # Sample to account for max number of publications we want to retrieve
        if len(ids) > n_pubs_max:
            ids = np.random.choice(ids, n_pubs_max, replace=False)

        print(f"Expansion will include {len(ids)} new publications.")

        # Retrieve publications from ids using a librarian
        # breakpoint()
        new_publications = self.librarian.get_publications(ids)

        # New atlas
        atl_exp = Atlas(new_publications)

        # Update the new atlas
        atl_exp.publications.update(atl.publications)
        atl_exp.projection = atl.projection

        return atl_exp

    ######################################################################
    # Filter Atlas
    ######################################################################

    def filter(
        self,
        atl: Atlas,
        attributes: list = [
            "abstract",
            "publication_date",
        ],
    ) -> Atlas:
        """Update an atlas by dropping publications (and corresponding data in projection) when certain fields are empty.

        Args:
            atl: the Atlas containing publications to filter

            attributes: the list of attributes to filter publications from the atlas if any of items are None for a publication. For example, if attributes = ["abstract"], then all publications `pub` such that `pub.abstract is None` is True will be removed from the atlas, along with the corresponding data in the projection.

        Returns:
            the filtered atlas
        """
        # Filter publications
        invalid_pubs = {
            id: pub
            for id, pub in atl.publications.items()
            if all([getattr(pub, attr) is None for attr in attributes])
        }
        # Do not update if unnecessary
        if not len(invalid_pubs):
            return atl

        filter_ids = invalid_pubs.keys()
        # Remove embeddings, ids from projection
        if atl.projection is not None and len(atl.projection):
            filter_indices = set()
            idx_to_id_new = []
            # From indexing
            for idx, id in enumerate(atl.projection.index_to_identifier):
                if id in filter_ids:
                    filter_indices.add(idx)
                else:
                    idx_to_id_new.append(id)
            # From embeddings
            embeddings = np.array(
                [
                    embedding
                    for idx, embedding in enumerate(atl.projection.embeddings)
                    if idx not in filter_indices
                ]
            )
            # From identifier to index map
            id_to_idx_new = {id: idx for idx, id in enumerate(idx_to_id_new)}
            # Overwrite Projection
            atl.projection = Projection(
                identifier_to_index=id_to_idx_new,
                index_to_identifier=idx_to_id_new,
                embeddings=embeddings,
            )

        # Remove publications
        atl.publications = {
            id: pub for id, pub in atl.publications.items() if id not in filter_ids
        }

        return atl
