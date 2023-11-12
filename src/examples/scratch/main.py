from examples.scratch import util

from sciterra.mapping.atlas import Atlas
from sciterra.mapping.cartography import Cartographer, iterate_expand
from sciterra.librarians import ADSLibrarian, SemanticScholarLibrarian
from sciterra.librarians import ADSLibrarian
from sciterra.vectorization import vectorizers

librarians = {
    "S2": SemanticScholarLibrarian(),
    "ADS": ADSLibrarian(),
}


def main(args):
    seed = args.seed
    target = args.target_size
    n_pubs_max = args.max_pubs_per_expand
    call_size = args.call_size
    max_failures = args.max_failed_expansions
    centered = args.centered
    librarian = librarians[args.api]
    vectorizer = vectorizers[args.vectorizer]
    bibtex_fp = args.bibtex_fp
    atlas_dir = args.atlas_dir

    util.set_seed(seed)

    crt = Cartographer(
        librarian=librarian,
        vectorizer=vectorizer(
            device="mps",
            model_path=args.model_path,
        ),
    )

    # # Get center from file
    atl_center = crt.bibtex_to_atlas(bibtex_fp)

    if centered:
        # center must be the sole publication in bibtex file
        (pub,) = list(atl_center.publications.values())
        center = pub.identifier
    else:
        center = None

    # Load
    atl = Atlas.load(atlas_dir)
    if len(atl):
        print(
            f"Loaded atlas has {len(atl)} publications and {len(atl.projection)} embeddings."
        )
    else:
        print(f"Initializing atlas.")
        atl = atl_center

    iterate_expand(
        atl=atl,
        crt=crt,
        atlas_dir=atlas_dir,
        target_size=target,
        max_failed_expansions=max_failures,
        center=center,
        n_pubs_max=n_pubs_max,
        call_size=call_size,
        record_pubs_per_update=True,
    )


if __name__ == "__main__":
    args = util.get_args()

    main(args)
