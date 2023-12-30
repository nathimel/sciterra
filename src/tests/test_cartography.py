"""Test basic expansion functionality with Cartographer, which has some convenient wrappers around each API librarian."""

import bibtexparser

import numpy as np

from datetime import datetime

from sciterra.mapping.atlas import Atlas
from sciterra.mapping.cartography import (
    Cartographer,
    pub_has_attributes,
    pub_has_fields_of_study,
)
from sciterra.librarians.s2librarian import SemanticScholarLibrarian
from sciterra.mapping.publication import Publication
from sciterra.vectorization import SciBERTVectorizer, Word2VecVectorizer

bib_dir = "src/tests/data/bib"
single_pub_bibtex_fp = f"{bib_dir}/single_publication.bib"
ten_pub_bibtex_fp = f"{bib_dir}/ten_publications.bib"
realistic_bibtex_fp = f"{bib_dir}/rdsg.bib"

##############################################################################
# SemanticScholar x SciBERT
##############################################################################

atlas_dir = "atlas_tmpdir"

# NOTE: Any time you are querying an API for papers, it is not a good idea to have strict tests on the resulting output size, since there is a significant amount of things out of our control, including that the online database may have added new papers.


class TestS2BibtexToAtlas:
    """Test functionality that maps a bibtex file to a list of identifiers, and then populates an atlas. The purpose of this functionality is to map a human-readable / very popular dataformat to the Atlas data structure."""

    librarian = SemanticScholarLibrarian()
    crt = Cartographer(librarian)

    def test_bibtex_to_atlas_single(self, tmp_path):
        # Load single file from bibtex
        # Load expected values
        bibtex_fp = single_pub_bibtex_fp
        with open(bibtex_fp, "r") as f:
            bib_database = bibtexparser.load(f)
        entry: dict = bib_database.entries[0]

        path = tmp_path / atlas_dir
        path.mkdir()
        # Construct Atlas
        atl = TestS2SBProjection.crt.bibtex_to_atlas(bibtex_fp)

        (pub,) = list(atl.publications.values())

        assert pub.identifier
        assert pub.abstract
        assert pub.publication_date
        assert pub.citation_count >= 0
        assert len(pub.citations) >= 0
        assert len(pub.references) >= 0

        assert entry["doi"] == pub.doi

    def test_bibtex_to_atlas_ten(self, tmp_path):
        # Load ten files from bibtex
        # Load expected values
        bibtex_fp = ten_pub_bibtex_fp
        with open(bibtex_fp, "r") as f:
            bib_database = bibtexparser.load(f)

        dois = [entry["doi"] for entry in bib_database.entries if "doi" in entry]

        path = tmp_path / atlas_dir
        path.mkdir()
        # Construct Atlas
        atl = TestS2SBProjection.crt.bibtex_to_atlas(bibtex_fp)

        for id, pub in atl.publications.items():
            assert pub.identifier == id
            assert pub.abstract
            assert pub.publication_date
            assert pub.citation_count >= 0
            assert len(pub.citations) >= 0
            assert len(pub.references) >= 0
            assert pub.doi in dois if hasattr(pub, "doi") else True

    def test_bibtex_to_atlas_realistic(self, tmp_path):
        # Load ten files from bibtex
        # Load expected values
        bibtex_fp = realistic_bibtex_fp
        with open(bibtex_fp, "r") as f:
            bib_database = bibtexparser.load(f)

        path = tmp_path / atlas_dir
        path.mkdir()
        # Construct Atlas
        atl = TestS2SBProjection.crt.bibtex_to_atlas(
            bibtex_fp,
            # multiprocess=False,
        )

        for id, pub in atl.publications.items():
            assert pub.identifier == id
            assert pub.abstract
            assert pub.publication_date
            assert pub.citation_count >= 0
            assert len(pub.citations) >= 0
            assert len(pub.references) >= 0

        # I find that I get 28 out of 86 total refs, i.e. less than a third of papers targeted.
        # or 32 lol
        # or 31
        # assert len(atl) == 28
        # assert len(atl) == 32
        # assert len(atl) == 31


class TestS2SBProjection:
    librarian = SemanticScholarLibrarian()
    vectorizer = SciBERTVectorizer()
    crt = Cartographer(librarian, vectorizer)

    def test_empty_projection(self):
        pubs = [Publication({"identifier": f"id_{i}"}) for i in range(10)]
        atl = Atlas(pubs)

        atl_proj = TestS2SBProjection.crt.project(atl)
        assert atl_proj.projection is None  # was filtered

    def test_dummy_projection_no_date(self):
        pubs = [
            Publication({"identifier": f"id_{i}", "abstract": "blah blah blah"})
            for i in range(10)
        ]
        atl = Atlas(pubs)

        atl_proj = TestS2SBProjection.crt.project(atl)
        assert all([hasattr(pub, "abstract") for pub in atl.publications.values()])
        assert atl_proj.projection is None  # was filtered

    def test_dummy_projection_no_abstract(self):
        pubs = [
            Publication(
                {"identifier": f"id_{i}", "publication_date": datetime(2023, 1, 1)}
            )
            for i in range(10)
        ]
        atl = Atlas(pubs)

        # breakpoint()
        atl_proj = TestS2SBProjection.crt.project(atl)

        assert all(
            [hasattr(pub, "publication_date") for pub in atl.publications.values()]
        )
        assert atl_proj.projection is None  # was filtered

    def test_dummy_projection(self):
        pubs = [
            Publication(
                {
                    "identifier": f"id_{i}",
                    "abstract": "blah blah blah",
                    "publication_date": datetime(2023, 1, 1),
                    "fields_of_study": ["dummy_field"],
                }
            )
            for i in range(10)
        ]
        atl = Atlas(pubs)

        atl_proj = TestS2SBProjection.crt.project(atl)

        projection = atl_proj.projection

        vector0 = projection.identifiers_to_embeddings(["id_0"])
        vector1 = projection.identifiers_to_embeddings(["id_9"])
        assert np.array_equal(vector0, vector1)

    def test_dummy_projection_partial(self):
        crt = Cartographer(vectorizer=Word2VecVectorizer())

        pubs = [
            Publication(
                {
                    "identifier": f"id_{0}",
                    "abstract": "We use cosmological hydrodynamic simulations with stellar feedback from the FIRE (Feedback In Realistic Environments) project to study the physical nature of Lyman limit systems (LLSs) at z ≤ 1.",  # everything here should be in the Word2Vec default vocab, since it trains on this abstract.
                    "publication_date": datetime(2023, 1, 1),
                    "fields_of_study": ["dummy_field"],
                }
            ),
            Publication(
                {
                    "identifier": f"id_{1}",
                    "abstract": "outofvocabularyitem",  # this should not
                    "publication_date": datetime(2023, 1, 1),
                }
            ),
            Publication(
                {
                    "identifier": f"id_{2}",
                    "abstract": "We use cosmological hydrodynamic simulations with stellar feedback from the FIRE (Feedback In Realistic Environments) project to study the physical nature of Lyman limit systems (LLSs) at z ≤ 1.",
                    "publication_date": datetime(2023, 1, 1),
                    "fields_of_study": ["dummy_field"],
                }
            ),
        ]
        atl = Atlas(pubs)

        atl_proj = crt.project(atl)

        assert len(atl_proj) == 2

    def test_single_projection(self, tmp_path):
        path = tmp_path / atlas_dir
        path.mkdir()
        # Construct Atlas
        atl = TestS2SBProjection.crt.bibtex_to_atlas(single_pub_bibtex_fp)

        atl_proj = TestS2SBProjection.crt.project(atl)
        projection = atl_proj.projection

        identifier = atl.ids()[0]
        assert projection.identifier_to_index == {identifier: 0}
        assert projection.index_to_identifier == (identifier,)
        assert projection.embeddings.shape == (1, 768)  # (num_pubs, embedding_dim)

    def test_project_correct_number(self, tmp_path):
        # Load single file from bibtex
        # Load expected values
        bibtex_fp = single_pub_bibtex_fp
        with open(bibtex_fp, "r") as f:
            bib_database = bibtexparser.load(f)

        path = tmp_path / atlas_dir
        path.mkdir()
        # Construct Atlas
        atl = TestS2SBProjection.crt.bibtex_to_atlas(bibtex_fp)

        pub = list(atl.publications.values())[0]
        ids = pub.citations + pub.references
        center = pub.identifier

        atl_exp_single = TestS2SBProjection.crt.expand(atl, center=center)
        atl_exp_single = TestS2SBProjection.crt.project(atl_exp_single)

        before = len(atl_exp_single)
        atl_exp_double = TestS2SBProjection.crt.expand(
            atl_exp_single, center=center, n_pubs_max=200
        )
        after = len(atl_exp_double)

        # Check that the second projection does not need to pull more docs than necessary

        # 1. Simulate first part of project
        # 'only project publications that have abstracts'
        atl_filtered = TestS2SBProjection.crt.filter_by_func(
            atl_exp_double,
            require_func=lambda pub: pub_has_attributes(
                pub,
                attributes=["abstract"],
            ),
        )

        # 'get only embeddings for publications not already projected in atlas'
        previously_embedded_ids = []
        if atl_filtered.projection is not None:
            previously_embedded_ids = atl_filtered.projection.identifier_to_index
        embed_ids = [
            id for id in atl_filtered.publications if id not in previously_embedded_ids
        ]

        # 2. Check that the number of abstracts to be embedded does not exceed the size of the previous expansion
        assert len(embed_ids) <= after - before


class TestS2SBExpand:
    librarian = SemanticScholarLibrarian()
    vectorizer = SciBERTVectorizer()
    crt = Cartographer(librarian, vectorizer)

    def test_expand_single(self, tmp_path):
        # Load single file from bibtex
        # Load expected values
        bibtex_fp = single_pub_bibtex_fp
        with open(bibtex_fp, "r") as f:
            bib_database = bibtexparser.load(f)

        path = tmp_path / atlas_dir
        path.mkdir()
        # Construct Atlas
        atl = TestS2SBExpand.crt.bibtex_to_atlas(bibtex_fp)

        pub = list(atl.publications.values())[0]
        ids = pub.citations + pub.references

        atl_exp = TestS2SBExpand.crt.expand(atl)

        assert len(atl_exp) > len(atl)
        # so far this holds, but things that aren't our fault could make it fail.
        assert len(atl_exp) == len(ids)

    def test_expand_double(self, tmp_path):
        # Load single file from bibtex
        # Load expected values
        bibtex_fp = single_pub_bibtex_fp
        with open(bibtex_fp, "r") as f:
            bib_database = bibtexparser.load(f)

        path = tmp_path / atlas_dir
        path.mkdir()
        # Construct Atlas
        atl = TestS2SBExpand.crt.bibtex_to_atlas(bibtex_fp)

        pub = list(atl.publications.values())[0]
        ids = pub.citations + pub.references

        atl_exp_single = TestS2SBExpand.crt.expand(atl)
        atl_exp_double = TestS2SBExpand.crt.expand(atl_exp_single, n_pubs_max=200)
        # empirically found this
        # note that all ids from atl_exp_single is 68282!
        assert len(atl_exp_double) == 200 + len(ids)

        # Save atlas
        atl_exp_double.save(path)

    def test_expand_center_single(self, tmp_path):
        # Load single file from bibtex
        # Load expected values
        bibtex_fp = single_pub_bibtex_fp
        with open(bibtex_fp, "r") as f:
            bib_database = bibtexparser.load(f)

        path = tmp_path / atlas_dir
        path.mkdir()
        # Construct Atlas
        atl = TestS2SBExpand.crt.bibtex_to_atlas(bibtex_fp)

        pub = list(atl.publications.values())[0]
        ids = pub.citations + pub.references
        center = pub.identifier

        atl_exp_single = TestS2SBExpand.crt.expand(atl, center=center)
        assert len(atl_exp_single) == len(ids)

        # Save atlas
        atl_exp_single.save(path)

    def test_expand_center_double(self, tmp_path):
        # Load single file from bibtex
        # Load expected values
        bibtex_fp = single_pub_bibtex_fp
        with open(bibtex_fp, "r") as f:
            bib_database = bibtexparser.load(f)

        path = tmp_path / atlas_dir
        path.mkdir()
        # Construct Atlas
        atl = TestS2SBExpand.crt.bibtex_to_atlas(bibtex_fp)

        pub = list(atl.publications.values())[0]
        ids = pub.citations + pub.references
        center = pub.identifier

        atl_exp_single = TestS2SBExpand.crt.expand(atl, center=center)
        atl_exp_single = TestS2SBExpand.crt.project(atl_exp_single)
        atl_exp_double = TestS2SBExpand.crt.expand(
            atl_exp_single, center=center, n_pubs_max=200
        )
        # empirically found this
        # do no assert len(atl_exp_double)  == 4000 + len(ids), because we want 4000 + len(valid_ids), i.e. 148 not 154
        # assert len(atl_exp_double) == 348 # why off by a few?
        # assert len(atl_exp_double) == 345

        atl_exp_double.save(path)


class TestTopography:
    librarian = SemanticScholarLibrarian()
    vectorizer = SciBERTVectorizer()
    crt = Cartographer(librarian, vectorizer)

    def test_measure_topography_full(self):
        bibtex_fp = ten_pub_bibtex_fp
        atl = TestTopography.crt.bibtex_to_atlas(bibtex_fp)
        atl = TestTopography.crt.project(atl)
        metrics = [
            "density",
            "edginess",
        ]
        measurements = TestTopography.crt.measure_topography(
            atl,
            metrics=metrics,
        )
        assert measurements.shape == tuple((len(atl), len(metrics)))

    def test_measure_topography_subset(self):
        bibtex_fp = ten_pub_bibtex_fp
        atl = TestTopography.crt.bibtex_to_atlas(bibtex_fp)
        atl = TestTopography.crt.project(atl)
        ids = atl.ids()[:-5]
        metrics = [
            "density",
            "edginess",
        ]
        measurements = TestTopography.crt.measure_topography(
            atl,
            ids=ids,
            metrics=metrics,
        )
        assert measurements.shape == tuple((len(ids), len(metrics)))

    def test_measure_topography_realistic(self):
        # Load single file from bibtex
        bibtex_fp = single_pub_bibtex_fp

        # Construct Atlas
        atl = TestTopography.crt.bibtex_to_atlas(bibtex_fp)

        pub = list(atl.publications.values())[0]
        ids = pub.citations + pub.references
        center = pub.identifier

        # Expand
        atl_exp_single = TestTopography.crt.expand(
            atl,
            center=center,
            n_pubs_max=200,
        )

        # Project, necessary for metrics!
        atl_exp_single = TestTopography.crt.project(atl_exp_single)
        ids = atl_exp_single.ids()

        metrics = [
            "density",
            "edginess",
        ]
        measurements = TestTopography.crt.measure_topography(
            atl_exp_single,
            ids=ids,
            metrics=metrics,
        )
        assert measurements.shape == tuple((len(ids), len(metrics)))


class TestConvergence:
    librarian = SemanticScholarLibrarian()
    vectorizer = SciBERTVectorizer()
    crt = Cartographer(librarian, vectorizer)

    def test_record_update_history(self):
        # Construct Atlas
        bibtex_fp = ten_pub_bibtex_fp

        atl = TestConvergence.crt.bibtex_to_atlas(bibtex_fp)

        # Mock expansion/update history data
        history = [
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
        ]

        TestConvergence.crt.record_update_history(
            atl.ids(),
            pubs_per_update=history,
        )

        expected = np.array(
            [
                -2,
                -2,
                2,
                2,
                1,
                0,
                1,
                2,
                2,
                -2,
            ]
        )
        actual = TestConvergence.crt.update_history

        assert np.array_equal(expected, actual)

    def test_converged_kernel_size(self):
        # Construct Atlas
        bibtex_fp = ten_pub_bibtex_fp
        atl = TestConvergence.crt.bibtex_to_atlas(bibtex_fp)
        atl = TestConvergence.crt.project(atl)

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
            atl.ids(),  # it=3
        ]

        TestConvergence.crt.record_update_history(
            atl.ids(),
            input,
        )

        expected = np.array(
            [
                3,
                3,
                2,
                2,
                1,
                0,
                1,
                2,
                2,
                3,
            ]
        )
        actual = TestConvergence.crt.update_history

        assert np.array_equal(expected, actual)
        # mock center
        actual = TestConvergence.crt.converged_kernel_size(atl)

        expected = np.array(
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

        assert np.array_equal(actual, expected)

    def test_pubs_per_update_expand_consistent(self, tmp_path):
        # Load single file from bibtex
        # Load expected values
        bibtex_fp = single_pub_bibtex_fp
        with open(bibtex_fp, "r") as f:
            bib_database = bibtexparser.load(f)

        path = tmp_path / atlas_dir
        path.mkdir()
        # Construct Atlas
        atl = TestConvergence.crt.bibtex_to_atlas(bibtex_fp)

        pub = list(atl.publications.values())[0]
        center = pub.identifier

        # Expand repeatedly
        num_expansions = 10
        for _ in range(num_expansions):
            atl = TestConvergence.crt.expand(
                atl,
                center=center,
                n_pubs_max=10,
                record_pubs_per_update=True,
            )

        assert len(TestConvergence.crt.pubs_per_update) == num_expansions

        TestConvergence.crt.record_update_history()

        # need to project all pubs before kernel calculations!
        atl = TestConvergence.crt.project(atl)

        # test convergence calculations
        result = TestConvergence.crt.converged_kernel_size(atl)
