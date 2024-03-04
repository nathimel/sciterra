"""Convenience functionality for organized expansions of an Atlas."""

import numpy as np

from .atlas import Atlas
from .cartography import Cartographer
from ..librarians import librarians
from ..vectorization import vectorizers
from typing import Callable

##############################################################################
# Iterative expansion helper function
##############################################################################


def iterate_expand(
    atl: Atlas,
    crt: Cartographer,
    atlas_dir: str,
    target_size: int,
    max_failed_expansions: int = 2,
    convergence_func: Callable[[Atlas], bool] = lambda _: False,
    center: str = None,
    n_pubs_max: int = None,
    call_size: int = None,
    n_sources_max: int = None,
    record_pubs_per_update: bool = False,
    **project_kwargs,
) -> Atlas:
    """Build out an Atlas of publications, i.e. search for similar publications. This is done by iterating a sequence of [expand, save, project, save, track, save]. The convergence criterion is:

    `converged = len(atl) >= target_size or failures >= max_failed_expansions or convergence_func(atl)`

    Args:
        atl: the Atlas to expand

        crt: the Cartographer to use

        atlas_dir: the directory where Atlas binaries will be saved/loaded from

        target_size: stop iterating when we reach this number of publications in the Atlas

        max_failed_expansions: stop iterating when we fail to add new publications after this many successive iterations. Default is 2.

        convergence_func: a function taking an Atlas and returns True if the expansion loop should stop. This serves as an additional disjunctive convergence criterion besides `target_size` and `max_failed_expansions` to exit the expansion loop, which will be called at the convergence check after each expansion. For example, a redundant function to choose would be `lambda atl: len(atl) >= target_size`. By default returns False.

        center: (if given) center the search on this publication, preferentially searching related publications.

        n_pubs_max: maximum number of publications allowed in the expansion.

        call_size: maximum number of papers to call API for in one query; if less than `len(paper_ids)`, chunking will be performed.

        n_sources_max: maximum number of publications (already in the atlas) to draw references and citations from.

        record_pubs_per_update: whether to track all the publications that exist in the resulting atlas to `self.pubs_per_update`. This should only be set to `True` when you need to later filter by degree of convergence of the atlas.

        project_kwargs: keyword args propagated to every `Cartographer.project` call during iterate_expand; see `Cartographer.filter_by_func`.

    Returns:
        atl: the expanded Atlas
    """
    converged = False
    print_progress = lambda atl: print(  # view incremental progress
        f"Atlas has {len(atl)} publications and {len(atl.projection) if atl.projection is not None else 'None'} embeddings."
    )

    # Expansion loop
    failures = 0
    # Count previous iterations from loaded atlas as part of total
    its = len(atl.history["pubs_per_update"]) if atl.history is not None else 0
    while not converged:
        its += 1
        len_prev = len(atl)

        print(f"\nExpansion {its}\n-------------------------------")

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
        atl = crt.project(
            atl,
            verbose=True,
            record_pubs_per_update=record_pubs_per_update,
            **project_kwargs,
        )
        print_progress(atl)
        atl.save(atlas_dir)

        atl = crt.track(atl, calculate_convergence=True)
        atl.save(atlas_dir)

        if len_prev == len(atl):
            failures += 0
        else:
            failures = 0

        converged = (
            len(atl) >= target_size 
            or failures >= max_failed_expansions
            or convergence_func(atl)
            )

    print(f"Exiting loop.")
    atl = crt.track(atl, calculate_convergence=True)
    atl.save(atlas_dir)

    converged_dict = {
        "target_size": len(atl) >= target_size,
        "max_failed_expansions": failures >= max_failed_expansions,
        "convergence_func": convergence_func(atl),
    }
    print(f"Expansion loop exited with atlas size {len(atl)} after {its} iterations meeting criteria: {converged_dict}.")

    return atl


def search_converged_ids(
    atl: Atlas,
    num_pubs_added: int,
    kernel_size: int = 16,
) -> list[str]:
    """Get all publication ids such that they did not change neighborhood identity over the duration of the addition of the last `num_pubs_added` publications added to the atlas during previous `Cartographer.expand` calls.
    
    Args:
        atl: Atlas to search for converged publications

        num_pubs_added: the number of publications that we require to have been added to the Atlas in order to compute convergence, in order to compute the find the update index in the Atlas history to filter by `>= kernel_size`.

        kernel_size: the minimum required size of the neighborhood that we will require to not have changed, w.r.t. `cond_d`. Default is 16.

    Returns: 
        converged_pub_ids: a list of Publication identifiers corresponding to publications that have converged acording to the criterion.
    """
    kernels = atl.history['kernel_size'] # shape `(num_pubs, max_update)`
    max_update = kernels.shape[1] # total number of updates to an Atlas

    # Define the update index to be the update such that greater than or equal to `num_pubs_added` have been added between this update and the final one. 
    # This is clearly dependent on the size of the update; for example, if each update in the history added many publications, or if `num_pubs_added` is small, this update_ind might in practice be max_update-1. 
    update_ind = None
    history = atl.history["pubs_per_update"][:-1] # exclude the final update
    for reverse_idx, ids_at_update in enumerate(history[::-1]):
        num_diff_from_final = len(set(atl.ids) - set(ids_at_update))
        if num_diff_from_final >= num_pubs_added:
            update_ind = max_update - reverse_idx - 1
            break


    if update_ind is None:
        print(f"Failed to find an update index > 0 such that {num_pubs_added} were added between then and the final update ({max_update-1}).")
        update_ind = 0
    else:
        print(f"Between update {update_ind} and the final update ({max_update-1}) there were {num_diff_from_final} publications added to the Atlas.")

    # Get all publications that have not changed neighborhoods up to kernel_size for the last con_d updates
    converged_filter = kernels[:, update_ind] >= kernel_size
    ids = np.array(atl.projection.index_to_identifier)
    converged_pub_ids = ids[converged_filter]    

    criterion = dict(num_pubs_added=num_pubs_added, kernel_size=kernel_size)

    print(
        f"Convergence criterion {criterion.items()} (=> index {update_ind} out of {max_update} total updates) yields {len(converged_pub_ids)} ids out of {len(atl)} total ids."
    )

    return converged_pub_ids


class AtlasTracer:
    """Convenience data structure for bookkeeping expansions of an Atlas that reduces boilerplate and ensures an aligned update history between the Atlas and Cartographer."""

    def __init__(
        self,
        atlas_dir: str,
        atlas_center_bibtex: str,
        librarian_name: str,
        vectorizer_name: str,
        librarian_kwargs: dict = dict(),
        vectorizer_kwargs: dict = dict(),
    ) -> None:
        """Convenience wrapper data structure for tracked expansions, by aligning the history of a Cartographer with an Atlas.

        Args:
            atlas_dir: absolute path of the directory to save atlas data in, propogated to `Atlas.load` and `Atlas.save`

            atlas_center_bibtex: absolute path of the .bib file containing a single entry, which is the core, central publication, and this entry contains an identifier recognizable by the librarian corresponding to `librarian_name`.

            librarian_name: a str name of a librarian, one of `librarians.librarians.keys()`, e.g. 'S2' or 'ADS'.

            vectorizer_name: a str name of a vectorizer, one of `vectorization.vectorizers.keys()`, e.g. 'BOW' or 'SciBERT'.

            librarian_kwargs: keyword args propogated to a Librarian initialization; if values are `None` they will be omitted

            vectorizer_kwargs: keyword args propogated to a Vectorizer initialization; if values are `None` they will be omitted
        """
        ######################################################################
        # Initialize cartography tools
        ######################################################################

        # Get librarian
        librarian = librarians[librarian_name]
        librarian_kwargs = {k: v for k, v in librarian_kwargs.items() if v is not None}

        # Get vectorizer
        vectorizer = vectorizers[vectorizer_name]
        # Get vectorizer kwargs if they are not null in config
        vectorizer_kwargs = {
            k: v for k, v in vectorizer_kwargs.items() if v is not None
        }

        self.cartographer = Cartographer(
            librarian=librarian(
                **librarian_kwargs,
            ),
            vectorizer=vectorizer(
                **vectorizer_kwargs,
            ),
        )

        ######################################################################
        # Initialize/Load Atlas
        ######################################################################
        self.atlas_dir = atlas_dir
        # Load
        atl = Atlas.load(self.atlas_dir)
        if len(atl):
            print(
                f"Loaded atlas has {len(atl)} publications and {len(atl.projection) if atl.projection is not None else 'None'} embeddings.\n"
            )
            # Crucial step: align the history of crt with atl
            if atl.history is not None:
                self.cartographer.pubs_per_update = atl.history["pubs_per_update"]
                print(
                    f"Loaded atlas at expansion iteration {len(atl.history['pubs_per_update'])}."
                )
        else:
            print(f"Initializing atlas.")

            # Get the bibtex file containing the seed publication
            bibtex_fp = atlas_center_bibtex

            # Get center from file
            atl_center = self.cartographer.bibtex_to_atlas(bibtex_fp)
            atl_center = self.cartographer.project(atl_center)

            num_entries = len(atl_center.publications.values())
            if num_entries > 1:
                raise Exception(
                    f"To build out a centered atlas, the center is specified by loading a bibtex file with a single entry. Found {num_entries} entries in {bibtex_fp}"
                )

            # Set the atlas center
            atl = atl_center
            (atl.center,) = atl.publications.keys()

        self.atlas = atl
        self.atlas.save(atlas_dirpath=self.atlas_dir)

    def expand_atlas(
        self,
        target_size: int,
        **kwargs,
    ) -> None:
        """Start or continue the expansion of the Atlas by calling `iterate_expand` with aligned Cartographer and Atlas, by default centered on atl.center.

        Args:
            target_size: stop iterating expansion when Atlas contains this many publications; argument propagated to `iterate_expand`.

            kwargs: keyword args propagated to `iterate_expand`.
        """

        if "center" not in kwargs:
            kwargs["center"] = self.atlas.center

        iterate_expand(
            self.atlas,
            self.cartographer,
            self.atlas_dir,
            target_size,
            **kwargs,
        )
