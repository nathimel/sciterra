"""Functions for manipulating an atlas based on the document embeddings of the abstracts of its publications."""

import bibtexparser
import inspect
import warnings

import numpy as np

from . import topography
from .atlas import Atlas
from ..librarians.librarian import Librarian
from ..vectorization.vectorizer import Vectorizer
from ..vectorization.projection import Projection, merge, get_empty_projection
from ..misc.utils import get_verbose, custom_formatwarning

from functools import partial
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

warnings.formatwarning = custom_formatwarning

##############################################################################
# Helper function
##############################################################################


def batch_cospsi_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Batch-process a pairwise cosine similarity matrix between embeddings.

    In order to avoid memory errors (e.g. bus error, segfaults) resulting from too large arrays, we batch process the construction of the cospsi_matrix.

    Args:
        embeddings: a 1D numpy array of embeddings

    Returns:
        cosine_similarities: a 2D numpy array representing the pairwise cosine similarity between each embedding
    """
    batch_size = min(1000, len(embeddings))  # Define a batch size

    cosine_similarities = None
    print(
        f"computing cosine similarity for {len(embeddings)} embeddings with batch size {batch_size}."
    )
    for i in tqdm(range(0, len(embeddings), batch_size)):
        # Process batches to compute cosine similarity
        batch = embeddings[i : i + batch_size]
        if cosine_similarities is None:
            cosine_similarities = cosine_similarity(batch, embeddings)
        else:
            cosine_similarities = np.vstack(
                (cosine_similarities, cosine_similarity(batch, embeddings))
            )

    return cosine_similarities


##############################################################################
# Cartographer
##############################################################################


class Cartographer:
    """A basic wrapper for obtaining and updating atlas projections.

    `self.librarian`: the Librarian object used to query a bibliographic database API.
    `self.vectorizer`: the Vectorizer object used to get a document embedding for each abstract
    `self.pubs_per_update`: a list of lists of publication str ids, representing the publications that exist at each time step / expansion update.
    `self.update_history`: an np.array of ints representing when publications were added. A value of -2 indicates no record of being added.
    """

    def __init__(
        self,
        librarian: Librarian = None,
        vectorizer: Vectorizer = None,
    ) -> None:
        self.librarian = librarian
        self.vectorizer = vectorizer

        self.pubs_per_update: list[list[str]] = []
        self.update_history: np.ndarray = None

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
        verbose = get_verbose(kwargs)

        with open(bibtex_fp, "r") as f:
            bib_database = bibtexparser.load(f)

        # Retrieve the identifier from each bibtex entry
        identifiers = [
            self.librarian.bibtex_entry_identifier(entry)
            for entry in bib_database.entries
        ]
        identifiers = [id for id in identifiers if id is not None]
        if len(identifiers) < len(bib_database.entries) and verbose:
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
        if len(publications) < len(identifiers) and verbose:
            warnings.warn(
                f"Only obtained {len(publications)} publications out of {len(identifiers)} total due to querying-related errors or missing abstracts."
            )

        # Construct atlas
        atl = Atlas(publications)
        return atl

    ######################################################################
    # Project Atlas
    ######################################################################

    def project(self, atl: Atlas, **kwargs) -> Atlas:
        """Update an atlas with its projection, i.e. the document embeddings for all publications using `self.vectorizer`, removing publications with no abstracts.

        Args:
            atl: the Atlas containing publications to project to document embeddings

        Returns:
            the updated atlas containing all nonempty-abstract-containing publications and their projection
        """
        verbose = get_verbose(kwargs)

        # Only project publications that have abstracts and publication dates
        atl_filtered = self.filter(atl)
        num_empty = len(atl) - len(atl_filtered)
        if num_empty and verbose:
            warnings.warn(
                f"{num_empty} publications were filtered due to missing crucial data. There are now {len(atl_filtered.bad_ids)} total ids that will be excluded in the future."
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

        if embed_ids:
            if verbose:
                if atl_filtered.projection is not None:
                    warnings.warn(
                        f"Found {len(atl_filtered.publications) - len(atl_filtered.projection)} publications not contained in Atlas projection."
                    )
                warnings.warn(f"Embedding {len(embed_ids)} total documents.")
            # Embed documents
            embeddings = self.vectorizer.embed_documents(
                [atl_filtered[id].abstract for id in embed_ids]
            )

        if embeddings is None and verbose:
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
        assert not set(atl_filtered.publications.keys()) - set(
            embedded_publications.keys()
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
        *args,
        center: str = None,
        n_pubs_max: int = 4000,
        n_sources_max: int = None,
        record_pubs_per_update: bool = False,
        **kwargs,
    ) -> Atlas:
        """Expand an atlas by retrieving a list of publications resulting from traversal of the citation network.

        Args:
            atl: the atlas containing the region to expand

            center: (if given) center the search on this publication, preferentially searching related publications.

            n_pubs_max: maximum number of publications allowed in the expansion.

            n_sources_max: maximum number of publications (already in the atlas) to draw references and citations from.

            record_pubs_per_update: whether to track all the publications that exist in the resulting atlas to `self.pubs_per_update`. This should only be set to `True` when you need to later filter by degree of convergence of the atlas.

        Returns:
            atl_expanded: the expanded atlas
        """
        existing_keys = set(atl.publications.keys())
        expand_keys = existing_keys
        if center is not None:
            # If atlas is initial
            if atl.projection is None:
                atl = self.project(atl)

            if len(atl.projection):
                # build cosine similarity matrix, of shape (1, num_pubs)
                cospsi_matrix = cosine_similarity(
                    atl.projection.identifiers_to_embeddings([center]),
                    atl.projection.embeddings,
                )
                # get most similar keys from center, including center itself
                sort_inds = np.argsort(cospsi_matrix)[::-1][
                    0
                ]  # argsort orders from least to greatest similarity, so reverse
                expand_keys = atl.projection.indices_to_identifiers(sort_inds)

        if n_sources_max is not None:
            expand_keys = expand_keys[:n_sources_max]

        # Get identifiers for the expansion
        # For each publication corresponding to an id in `expand_keys`, collect the ids corresponding to the publication's references and citations.
        ids = set()
        for key in expand_keys:
            ids_i = set(atl[key].references + atl[key].citations)
            # Prune for obvious overlap, and for ids that have previously failed
            ids.update(ids_i - existing_keys - atl.bad_ids)
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
        new_publications = self.librarian.get_publications(ids, *args, **kwargs)

        # New atlas
        atl_exp = Atlas(new_publications)

        # Update the new atlas
        atl_exp.publications.update(atl.publications)
        atl_exp.bad_ids = atl.bad_ids
        atl_exp.projection = (
            atl.projection
        )  # new projection will be updated in `project`

        # Record the new list of publications
        if record_pubs_per_update:
            self.pubs_per_update.append(list(atl_exp.publications.keys()))

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
            if (pub is None or any([getattr(pub, attr) is None for attr in attributes]))
        }
        # Do not update if unnecessary
        if not len(invalid_pubs):
            return atl

        filter_ids = invalid_pubs.keys()

        # Keep track of the bad identifiers to skip them in future expansions
        new_bad_ids = atl.bad_ids.union(filter_ids)

        # Filter embeddings, ids from projection
        # if len(atl.projection):
        if atl.projection is None:
            new_projection = None
        else:
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
            # Construct new, filtered projection
            new_projection = Projection(
                identifier_to_index=id_to_idx_new,
                index_to_identifier=idx_to_id_new,
                embeddings=embeddings,
            )

        # Keep only filtered publications
        new_publications = [
            pub for id, pub in atl.publications.items() if id not in filter_ids
        ]

        # Construct new atlas
        atl_filtered = Atlas(new_publications, new_projection)
        atl_filtered.bad_ids = new_bad_ids
        return atl_filtered

    ########################################################################
    # Record Atlas history
    ########################################################################

    def track(
        self,
        atl: Atlas,
        pubs: list[str] = None,
        pubs_per_update: list[list[str]] = None,
    ) -> Atlas:
        """Overwrite the data associated with tracking degree of convergence of publications in an atlas over multiple expansions. N.B.: the atlas must be fully projected, or else `converged_kernel_size` will raise a KeyError.

        Args:
            atl: the Atlas that will be updated by overwriting `Atlas.history`
        """
        self.record_update_history(pubs, pubs_per_update)
        kernel_size = self.converged_kernel_size(atl)
        atl.history = {
            "pubs_per_update": self.pubs_per_update
            if pubs_per_update is None
            else pubs_per_update,
            "kernel_size": kernel_size,
        }
        return atl

    ########################################################################
    # Record Atlas history
    ########################################################################

    def record_update_history(
        self,
        pubs: list[str] = None,
        pubs_per_update: list[list[str]] = None,
    ) -> None:
        """Record when publications were added, by updating atl.update_history.

        atl.update_history is a np.array of ints representing when publications were added. A value of -2 indicates no record of being added.

        Args:
            pubs: a list of str ids corresponding to publications at the final update in the update history. By default `None`, and `self.pubs_per_update[-1]` will be used.

            pubs_per_update: a list of which publications existed at which iteration, with the index of the overall list corresponding to the iteration the publication was added. By default `None`, and `self.pubs_per_update` will be used.

        Updates:
            `self.update_history`: an np.array of ints representing when publications were added. A value of -2 indicates no record of being added.

        Returns:
            `None`
        """
        if pubs is None:
            pubs = self.pubs_per_update[-1]

        if pubs_per_update is None:
            pubs_per_update = self.pubs_per_update

        # Loop backwards
        i_max = len(pubs_per_update) - 1
        update_history = np.full(len(pubs), -2)
        for i, pubs_i in enumerate(pubs_per_update[::-1]):
            is_in = np.isin(pubs, pubs_i)
            update_history[is_in] = i_max - i
        self.update_history = update_history

    ########################################################################
    # Calculate Atlas convergence
    ########################################################################

    def converged_kernel_size(self, atl: Atlas) -> np.ndarray:
        """Calculate the largest size of the kernel that's converged (at differing levels of convergence) for each publication in a sample at each update.

        Args:
            atl: Atlas containing publications; for each publication we compute the largest converged kernel size at each update

        Returns:
            kernel_size: an array of ints of shape `(num_pubs, max_update)` representing the kernel size for converged kernels.
                - The first column indicates the largest kernel size that hasn't changed since the beginning,
                - The second column indicates the largest kernel size that hasn't changed since the first update,
                - etc. for the nth column.
        """

        if self.update_history is None:
            raise ValueError(
                "update_history is None; make sure you have called record_update_history()!"
            )

        if -2 in self.update_history:
            raise ValueError(
                "Incomplete update history as indicated by entries with values of -2."
            )

        publications = np.array(list(atl.publications.keys()))

        # 1. Loop over each publication
        cospsi_kernel = []
        for pub in tqdm(publications):
            # 2. Identify the similarity with the other publications relative to this publication, and sort accordingly.
            cospsi = cosine_similarity(
                atl.projection.identifiers_to_embeddings([pub]),
                atl.projection.embeddings,
            ).flatten()  # shape `(num_pubs,)`
            sort_inds = np.argsort(cospsi)[::-1]  # shape `(num_pubs,)`

            # 3. Identify the expansion iteration at which those publications were added to the atlas (`sorted_history`).
            sorted_history = self.update_history[sort_inds]  # shape `(num_pubs,)`

            # 4. Identify the latest iteration at which any publication was added to the atlas; this can be less than the total iterations.
            last_update = self.update_history.max()

            # 5. Loop through each iteration until `last_update`, and identify which publications were added at or before that iteration.
            result_2 = [
                # 6. Compute how many publications out we can go and still only contain publications added at or before that iteration.
                # Use `argmin` to get the first instance of False
                # Finally, subtract 1: we want the first index before False.
                np.argmin(sorted_history <= update) - 1
                for update in range(last_update)
            ]  # shape `(num_pubs, last_update)`

            cospsi_kernel.append(result_2)

        return np.array(cospsi_kernel)

    ########################################################################
    # Measure Atlas topography
    ########################################################################

    def measure_topography(
        self,
        atl: Atlas,
        publication_indices: np.ndarray = None,
        metrics: list[str] = ["density"],
        min_prior_pubs: int = 2,
        kernel_size=16,
        **kwargs,
    ):
        """Measure topographic properties of all publications relative to prior
        publications.

        Args:

            atl: the Atlas to measure

            publication_indices: an np.ndarray of ints representing the indices of publications in the Atlas projection to measure

            metrics: A list of strings representing the metrics to use. Options are...
                constant_asymmetry: The asymmetry of a publication $p_i$ w.r.t the entire atlas $\\{ p_j \\forall j \\in \\{1, ..., k\\} \\} where $k$ is the length of the atlas

                    $| \\sum_{j}^{k-1}( p_i - p_j ) |$

                kernel_constant_asymmetry: The asymmetry of a publication w.r.t. its kernel, { p_j for all j in {1, ..., k} } where k is `kernel_size`, i.e. the k nearest neighbors.

                density: the density of a publication's surrounding area, estimated by a heuristic inspired by mass / volume = k publications divided by the minimum arc length enclosing the furthest publication.

                    $\\frac{ k }{ smoothing\\_length(k) }$

                smoothing_length: The distance (in radians) to the farthest publication in the kernel, i.e. the kth nearest neighbor.

            min_prior_pubs: The minimum number of publications prior to the target publication for which to calculate the metric.

            kernel_size: the number of publications surrounding the publication for which to compute the topography metric, i.e. k nearest neighbors for k=kernel_size.

        Returns:
            estimates: an np.ndarray of shape `(len(publication_indices), len(metrics))` representing the estimated topography metric values for each publication.
        """
        verbose = get_verbose(kwargs)

        # By default calculate for all publications
        if publication_indices is None:
            publication_indices = np.array(
                list(atl.projection.identifier_to_index.values())
            )

        # Get publication dates, for filtering
        dates = np.array(
            [
                atl.publications[
                    atl.projection.index_to_identifier[idx]
                ].publication_date
                for idx in publication_indices
            ]
        )

        cospsi_matrix = batch_cospsi_matrix(atl.projection.embeddings)

        print(f"Computing {metrics} for {len(publication_indices)} publications.")
        estimates = []
        for idx in tqdm(publication_indices):
            # Get the date of publication
            identifier = atl.projection.index_to_identifier[idx]
            date = atl[identifier].publication_date

            # Identify prior publications
            is_prior = dates < date
            if is_prior.sum() < min_prior_pubs:
                estimates.append([np.nan for _ in metrics])
                continue

            # Choose valid publications
            is_other = publication_indices != idx
            is_valid = is_prior & is_other
            valid_indices = publication_indices[is_valid]

            kwargs = {
                "idx": idx,
                "cospsi_matrix": cospsi_matrix,
                "valid_indices": valid_indices,
                "publication_indices": publication_indices,
                "embeddings": atl.projection.embeddings,
                "kernel_size": kernel_size,
            }

            def call_metric(
                metric: str,
                **kwargs,
            ) -> float:
                """Wrapper function to simplify topography metric api."""
                # Get the metric
                fn = getattr(topography, f"{metric}_metric")

                # Identify arguments to pass
                fn_args = inspect.getfullargspec(fn)
                used_kwargs = {}
                for key, value in kwargs.items():
                    if key in fn_args.args:
                        used_kwargs[key] = value
                # Call
                estimate = fn(**used_kwargs)
                return estimate

            estimates.append([call_metric(metric, **kwargs) for metric in metrics])
        estimates = np.array(estimates)

        return estimates


##############################################################################
# Iterative expansion helper function
##############################################################################


def iterate_expand(
    atl: Atlas,
    crt: Cartographer,
    atlas_dir: str,
    target_size: int,
    max_failed_expansions: int,
    center: str = None,
    n_pubs_max: int = None,
    call_size: int = None,
    n_sources_max: int = None,
    record_pubs_per_update: bool = False,
) -> None:
    """Build out an Atlas of publications, i.e. search for similar publications. This is done by iterating a sequence of [expand, save, project, save, track, save].

    Args:
        atl: the Atlas to expand

        crt: the Cartographer to use

        atlas_dir: the directory where Atlas binaries will be saved/loaded from

        target_size: stop iterating when we reach this number of publications in the Atlas

        max_failed_expansions: stop iterating when we fail to add new publications after this many successive iterations.

        center: (if given) center the search on this publication, preferentially searching related publications.

        n_pubs_max: maximum number of publications allowed in the expansion.

        call_size: maximum number of papers to call API for in one query; if less than `len(paper_ids)`, chunking will be performed.

        n_sources_max: maximum number of publications (already in the atlas) to draw references and citations from.

        record_pubs_per_update: whether to track all the publications that exist in the resulting atlas to `self.pubs_per_update`. This should only be set to `True` when you need to later filter by degree of convergence of the atlas.

    """
    converged = False
    print_progress = lambda atl: print(  # view incremental progress
        f"Atlas has {len(atl)} publications and {len(atl.projection) if atl.projection is not None else 'None'} embeddings."
    )

    # Expansion loop
    failures = 0
    while not converged:
        len_prev = len(atl)

        # Retrieve up to n_pubs_max citations and references.
        atl = crt.expand(
            atl,
            center=center,
            n_pubs_max=n_pubs_max,
            call_size=call_size,
            n_sources_max=n_sources_max,
            record_pubs_per_update=record_pubs_per_update,
        )
        print_progress(atl)
        atl.save(atlas_dir)

        # Obtain document embeddings for all new abstracts.
        atl = crt.project(atl, verbose=True)
        print_progress(atl)
        atl.save(atlas_dir)

        atl = crt.track(atl)
        atl.save(atlas_dir)

        if len_prev == len(atl):
            failures += 0
        else:
            failures = 0

        converged = len(atl) >= target_size or failures >= max_failed_expansions
        print()

    print(f"Expansion loop exited with atlas size {len(atl)}.")
