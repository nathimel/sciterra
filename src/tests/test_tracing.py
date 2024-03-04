import numpy as np

from sciterra.mapping.cartography import Cartographer
from sciterra.mapping.tracing import iterate_expand, AtlasTracer, search_converged_ids
from sciterra.librarians.s2librarian import SemanticScholarLibrarian
from sciterra.vectorization import SciBERTVectorizer

from .test_cartography import (
    single_pub_bibtex_fp,
    atlas_dir,
    ten_pub_bibtex_fp,
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
            vectorizer_kwargs={
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

    def test_atlas_tracer_with_convergence_func(self, tmp_path):
        path = tmp_path / atlas_dir
        path.mkdir()

        tracer = AtlasTracer(
            path,
            single_pub_bibtex_fp,
            "S2",
            "Word2Vec",
            vectorizer_kwargs={
                "corpus_path": astro_corpus_1,
                "model_path": model_path_1,
            },
        )

        # Simple dummy convergence criterion: length is divisible by 3
        func = lambda atl: len(atl) % 3 == 0

        tracer.expand_atlas(
            target_size=100,
            max_failed_expansions=2,
            convergence_func=func,
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
            vectorizer_kwargs={
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


class TestSearchConvergence:
    librarian = SemanticScholarLibrarian()
    vectorizer = SciBERTVectorizer()
    crt = Cartographer(librarian, vectorizer)

    # Construct Atlas
    bibtex_fp = ten_pub_bibtex_fp
    atl = crt.bibtex_to_atlas(bibtex_fp)
    atl = crt.project(atl)

    # Mock expansion/update history data
    input = [
        ["f2c251056dee4c6f9130b31e5e3e4b3296051c49"],  # it=0
        [
            "4364af31229f7e9a3d83a289a928b2f2a43d30cb",  # it=1
            "f2c251056dee4c6f9130b31e5e3e4b3296051c49",
            "287fa946f30eaa78ea86f9c5bd61d67238202356",
        ],
        [
            "50dea78a96f03ba7fc3398c5deea5174630ef186",  # it=2
            "54a83cd1d94814b0f37ee48084260a2d1882648d",
            "4364af31229f7e9a3d83a289a928b2f2a43d30cb",
            "f2c251056dee4c6f9130b31e5e3e4b3296051c49",
            "287fa946f30eaa78ea86f9c5bd61d67238202356",
            "2e6438be4901cb9b42ff23dcc3d433789b37d032",
            "04da6471743468b6bb1d26dd9a6eac4c03ca73ee",
        ],
        # it=3, len=10. This is equiv to atl.ids
        [
            "9d1a233164f27342d316662821e9a6bb855c25b4",
            "af6a1c9da102e29fee5d309ec33831207e9f23e5",
            # ----- it 2 ----
            "50dea78a96f03ba7fc3398c5deea5174630ef186",
            "54a83cd1d94814b0f37ee48084260a2d1882648d",
            "4364af31229f7e9a3d83a289a928b2f2a43d30cb",
            "f2c251056dee4c6f9130b31e5e3e4b3296051c49",
            "287fa946f30eaa78ea86f9c5bd61d67238202356",
            "2e6438be4901cb9b42ff23dcc3d433789b37d032",
            "04da6471743468b6bb1d26dd9a6eac4c03ca73ee",
            # ----- end it 2 ----
            "0def4f553107451204b34470890d019b373798b5",
        ],
    ]

    crt.record_update_history(
        atl.ids,
        input,
    )

    kernel_size_arr = np.array(
        [
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, 0],
            [-1, -1, 1],
            [-1, 0, 3],
            [0, 0, 2],
            [-1, 0, 0],
            [-1, -1, 1],
            [-1, -1, 3],
            [-1, -1, -1],
        ]
    )

    # Simulates crt.track
    atl.history = {
        "pubs_per_update": input,
        "kernel_size": kernel_size_arr,
    }

    def test_search_converged_ids_final_column_nontrivial(self):
        atl = TestSearchConvergence.atl
        # Now compute the number of converged publications s.t. kernel_size > 2, starting at the update index over which 3 publications were added. This should give us the last index, since 3 were added then, and exactly 3 'converged' publications, since the last column of `expected` has 3 pubs with values >= 2

        # First test that we get the expected result when indexing
        # select the ids that have k=3,2, and 3 in last col of `expected`
        expected_ids = [x for i, x in enumerate(atl.ids) if i in [4, 5, 8]]

        actual_conv_ids = search_converged_ids(
            atl,
            num_pubs_added=3,
            kernel_size=2,
        )
        actual_sorted = sorted(actual_conv_ids)
        expected_sorted = sorted(expected_ids)

        assert actual_sorted == expected_sorted

    def test_search_converged_ids_final_column_trivial(self):
        # Now test we get the same thing with num_pubs added =0
        atl = TestSearchConvergence.atl
        expected_ids = [x for i, x in enumerate(atl.ids) if i in [4, 5, 8]]

        actual_conv_ids = search_converged_ids(
            atl,
            num_pubs_added=0,
            kernel_size=2,
        )
        actual_sorted = sorted(actual_conv_ids)
        expected_sorted = sorted(expected_ids)
        assert actual_sorted == expected_sorted

    def test_search_converged_ids_first_column_empty(self):
        atl = TestSearchConvergence.atl
        # Now test that when num_pubs_added is 100, this is greater than the total atlas size, so we should index the first update, and there will be 0 converged pubs, since kernel size = 2.
        expected_ids = []
        actual_conv_ids = search_converged_ids(
            atl,
            num_pubs_added=100,
            kernel_size=2,
        )
        actual_sorted = sorted(actual_conv_ids)
        expected_sorted = sorted(expected_ids)
        assert actual_sorted == expected_sorted

    def test_search_converged_ids_first_column_get_center(self):
        atl = TestSearchConvergence.atl
        # Now test the same condition, but only require kernel_size to be 0. Then we should still get one pub: the first pub added to the atlas.
        expected_ids = [
            "f2c251056dee4c6f9130b31e5e3e4b3296051c49",
        ]
        actual_conv_ids = search_converged_ids(
            atl,
            num_pubs_added=100,
            kernel_size=0,
        )
        actual_sorted = sorted(actual_conv_ids)
        expected_sorted = sorted(expected_ids)
        assert actual_sorted == expected_sorted
