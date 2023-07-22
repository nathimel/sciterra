import util

from sciterra.mapping.atlas import Atlas
from sciterra.mapping.cartography import Cartographer
from sciterra.librarians import ADSLibrarian, SemanticScholarLibrarian
from sciterra.librarians import ADSLibrarian
from sciterra.vectorization.scibert import SciBERTVectorizer

librarians = {
    'S2': SemanticScholarLibrarian(),
    'ADS': ADSLibrarian(),
}

def main(args):
    seed = args.seed
    target = args.target_size
    n_pubs_max = args.max_pubs_per_expand
    centered = args.centered
    librarian = librarians[args.api]
    bibtex_fp = args.bibtex_fp
    atlas_dir = args.atlas_dir

    util.set_seed(seed)

    crt = Cartographer(
        librarian=librarian,
        vectorizer=SciBERTVectorizer(device="mps"),
    )

    # # Get center from file
    atl_center = crt.bibtex_to_atlas(bibtex_fp)

    if centered:
        # center must be the sole publication in bibtex file
        pub, = list(atl_center.publications.values())
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

    converged = False
    print_progress = lambda atl: print(  # view incremental progress
        f"Atlas has {len(atl)} publications and {len(atl.projection) if atl.projection is not None else 'None'} embeddings."
    )

    # Expansion loop
    while not converged:
        len_prev = len(atl)

        # Retrieve up to n_pubs_max citations and references.
        atl = crt.expand(atl, center=center, n_pubs_max=n_pubs_max)
        print_progress(atl)
        atl.save(atlas_dir)

        # Obtain document embeddings for all new abstracts.
        atl = crt.project(atl, verbose=True)
        print_progress(atl)
        atl.save(atlas_dir)

        converged = len(atl) >= target or len_prev == len(atl)
        print()

    print(f"Expansion loop exited with atlas size {len(atl)}.")


if __name__ == "__main__":
    args = util.get_args()

    main(args)
