from sciterra.mapping.atlas import Atlas
from sciterra.mapping.cartography import Cartographer
from sciterra.librarians.s2librarian import SemanticScholarLibrarian
from sciterra.librarians.adslibrarian import ADSLibrarian
from sciterra.vectorization.scibert import SciBERTVectorizer

# TODO: Add commandline args
# bibtex_fp = "data/hafenLowredshiftLymanLimit2017.bib"
# bibtex_fp = "data/Imeletal2022a.bib"
bibtex_fp = "data/lewis1970.bib"
atlas_dir = "outputs/atlas_s2"


print_progress = lambda atl: print(
    f"Atlas has {len(atl)} publications and {len(atl.projection) if atl.projection is not None else 'None'} embeddings."
)


def main():
    crt = Cartographer(
        librarian=SemanticScholarLibrarian(),
        vectorizer=SciBERTVectorizer(device="mps"),
    )

    # # Get center from file
    atl_center = crt.bibtex_to_atlas(bibtex_fp)
    pub = list(atl_center.publications.values())[0]
    center = pub.identifier

    # No center
    # center = None

    # Load
    atl = Atlas.load(atlas_dir)
    if len(atl):
        print(
            f"Loaded atlas has {len(atl)} publications and {len(atl.projection)} embeddings."
        )
    else:
        print(f"Initializing atlas.")
        atl = atl_center

    # Expansion loop
    target = 10000
    while len(atl) < target:
        # breakpoint()
        atl = crt.expand(atl, center, n_pubs_max=1000)
        print_progress(atl)
        atl.save(atlas_dir)
        # TODO: if no new projections after n=2 times, exit.
        atl = crt.project(atl, verbose=True)
        print_progress(atl)
        atl.save(atlas_dir)
        print()

    print("Expansion target reached.")


if __name__ == "__main__":
    main()
