from sciterra.mapping.cartography import Cartographer
from sciterra.mapping.tracing import iterate_expand, AtlasTracer
from sciterra.librarians.s2librarian import SemanticScholarLibrarian
from sciterra.vectorization import SciBERTVectorizer

from .test_cartography import (
    single_pub_bibtex_fp, 
    atlas_dir,
)
from .test_vectorization import astro_corpus_1, model_path_1

class TestExpansion:
    def test_iterate_expand(self, tmp_path):
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


    def test_atlas_tracer_s2(self, tmp_path):

        path = tmp_path / atlas_dir
        path.mkdir()        

        tracer = AtlasTracer(
            path,
            single_pub_bibtex_fp,
            "S2",
            "Word2Vec",
            vectorizer_kwargs = {
                "corpus_path": astro_corpus_1,
                "model_path": model_path_1,
            },
        )
        tracer.expand_atlas(
            target_size=100,
            max_failed_expansions=2,
            n_pubs_max=10,
            call_size=None,
            n_sources_max=None,
            record_pubs_per_update=True,            
        )

    def test_atlas_tracer_ads(self, tmp_path):

        path = tmp_path / atlas_dir
        path.mkdir()        

        tracer = AtlasTracer(
            path,
            single_pub_bibtex_fp,
            "ADS",
            "BOW",
            vectorizer_kwargs = {
                "corpus_path": astro_corpus_1,
                "model_path": model_path_1,
            },
        )
        tracer.expand_atlas(
            target_size=100,
            max_failed_expansions=2,
            n_pubs_max=10,
            call_size=None,
            n_sources_max=None,
            record_pubs_per_update=True,            
        )