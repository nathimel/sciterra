"""Convenience function for building out an Atlas of publications via iterative expansion, i.e. search for similar publications."""

from .atlas import Atlas
from .cartography import Cartographer

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
    """Build out an Atlas of publications by iterating a sequence of [expand, save, project, save, track, save].
    
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
