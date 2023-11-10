from sciterra.mapping.search import iterate_expand
from sciterra.mapping.cartography import Cartographer
from sciterra.librarians.s2librarian import SemanticScholarLibrarian
from sciterra.vectorization.scibert import SciBERTVectorizer

single_pub_bibtex_fp = "src/tests/data/single_publication.bib"

atlas_dir = "atlas_tmpdir"

class TestSearch:

    def test_search(self, tmp_path):

        librarian = SemanticScholarLibrarian()
        vectorizer = SciBERTVectorizer()
        crt = Cartographer(librarian, vectorizer)

        # Load single file from bibtex
        bibtex_fp = single_pub_bibtex_fp

        path = tmp_path / atlas_dir
        path.mkdir()

        # Construct Atlas
        atl = crt.bibtex_to_atlas(bibtex_fp)

        pub = list(atl.publications.values())[0]
        center = pub.identifier        

        iterate_expand(
            atl=atl,
            crt=crt,
            atlas_dir=path,
            target_size=100,
            max_failed_expansions=2,
            center=center,
            n_pubs_max=10,
            call_size=None,
            n_sources_max=None,
            record_pubs_per_update=True,
        )
