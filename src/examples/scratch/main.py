from sciterra.atlas import Atlas
from sciterra.cartography import Cartographer
from sciterra.librarians.s2librarian import SemanticScholarLibrarian
from sciterra.vectorization.scibert import SciBERTVectorizer


bibtex_fp = "data/single_publication.bib"
save_path = "outputs/atlas_single"

def main():

    crt = Cartographer(
        librarian = SemanticScholarLibrarian(), 
        vectorizer = SciBERTVectorizer(),
    )

    # Construct Atlas
    atl = crt.bibtex_to_atlas(bibtex_fp)

    pub = list(atl.publications.values())[0]
    center = pub.identifier

    atl_exp_single = crt.expand(atl, center=center)

    atl_exp_single.save(save_path)

    atl_exp_loaded = Atlas.load(save_path)

    if not atl_exp_loaded.publications.keys() == atl_exp_single.publications.keys():
        breakpoint()

    breakpoint()


if __name__ == "__main__":
    main()