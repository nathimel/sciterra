"""Convenience functionality for organized expansions of an Atlas."""

from .atlas import Atlas
from .cartography import Cartographer
from ..librarians import librarians
from ..vectorization import vectorizers

##############################################################################
# Iterative expansion helper function
##############################################################################


def iterate_expand(
    atl: Atlas,
    crt: Cartographer,
    atlas_dir: str,
    target_size: int,
    max_failed_expansions: int = 2,
    center: str = None,
    n_pubs_max: int = None,
    call_size: int = None,
    n_sources_max: int = None,
    record_pubs_per_update: bool = False,
    **project_kwargs,
) -> Atlas:
    """Build out an Atlas of publications, i.e. search for similar publications. This is done by iterating a sequence of [expand, save, project, save, track, save].

    Args:
        atl: the Atlas to expand

        crt: the Cartographer to use

        atlas_dir: the directory where Atlas binaries will be saved/loaded from

        target_size: stop iterating when we reach this number of publications in the Atlas

        max_failed_expansions: stop iterating when we fail to add new publications after this many successive iterations. Default is 2.

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

        atl = crt.track(atl)
        atl.save(atlas_dir)

        if len_prev == len(atl):
            failures += 0
        else:
            failures = 0

        converged = len(atl) >= target_size or failures >= max_failed_expansions

    print("Calculating degree of convergence for all publications.")
    atl = crt.track(atl, calculate_convergence=True)
    atl.save(atlas_dir)

    print(f"Expansion loop exited with atlas size {len(atl)} after {its} iterations.")
    return atl


class AtlasTracer:
    """Convenience data structure for bookkeeping expansions of an Atlas that reduces boilerplate and ensures an aligned update history between the Atlas and Cartographer."""

    def __init__(
        self,
        atlas_dir: str,
        atlas_center_bibtex: str,
        librarian_name: str,
        vectorizer_name: str,
        vectorizer_kwargs: dict = None,
    ) -> None:
        """Convenience wrapper data structure for tracked expansions, by aligning the history of a Cartographer with an Atlas.

        Args:
            atlas_dir: absolute path of the directory to save atlas data in, propogated to `Atlas.load` and `Atlas.save`

            atlas_center_bibtex: absolute path of the .bib file containing a single entry, which is the core, central publication, and this entry contains an identifier recognizable by the librarian corresponding to `librarian_name`.

            librarian_name: a str name of a librarian, one of `librarians.librarians.keys()`, e.g. 'S2' or 'ADS'.

            vectorizer_name: a str name of a vectorizer, one of `vectorization.vectorizers.keys()`, e.g. 'BOW' or 'SciBERT'.

            vectorizer_kwargs: keyword args propogated to a Vectorizer initialization; if values are `None` they will be omitted
        """
        ######################################################################
        # Initialize cartography tools
        ######################################################################

        # Get librarian
        librarian = librarians[librarian_name]

        # Get vectorizer
        vectorizer = vectorizers[vectorizer_name]
        # Get vectorizer kwargs if they are not null in config
        v_kwargs = {k: v for k, v in vectorizer_kwargs.items() if v is not None}

        self.cartographer = Cartographer(
            librarian=librarian(),
            vectorizer=vectorizer(
                **v_kwargs,
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
