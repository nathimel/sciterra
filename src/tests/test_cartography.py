"""Test basic expansion functionality with Cartographer, which has some convenient wrappers around each API librarian."""

import bibtexparser

import numpy as np

from datetime import datetime

from sciterra.mapping.atlas import Atlas
from sciterra.mapping.cartography import Cartographer
from sciterra.librarians.s2librarian import SemanticScholarLibrarian
from sciterra.mapping.publication import Publication
from sciterra.vectorization.scibert import SciBERTVectorizer

single_pub_bibtex_fp = "src/tests/data/single_publication.bib"
ten_pub_bibtex_fp = "src/tests/data/ten_publications.bib"
realistic_bibtex_fp = "src/tests/data/rdsg.bib"

##############################################################################
# SemanticScholar x SciBERT
##############################################################################

atlas_dir = "atlas_tmpdir"


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
        # assert len(atl) == 28
        assert len(atl) == 32


class TestS2SBProjection:
    librarian = SemanticScholarLibrarian()
    vectorizer = SciBERTVectorizer()
    crt = Cartographer(librarian, vectorizer)

    def test_empty_projection(self):
        pubs = [Publication({"identifier": f"id_{i}"}) for i in range(10)]
        atl = Atlas(pubs)

        atl_proj = TestS2SBProjection.crt.project(atl)
        assert atl_proj.projection is None # was filtered

    def test_dummy_projection_no_date(self):
        pubs = [
            Publication({"identifier": f"id_{i}", "abstract": "blah blah blah"})
            for i in range(10)
        ]
        atl = Atlas(pubs)

        atl_proj = TestS2SBProjection.crt.project(atl)
        assert all([hasattr(pub, "abstract") for pub in  atl.publications.values()])
        assert atl_proj.projection is None # was filtered

    def test_dummy_projection_no_abstract(self):
        pubs = [
            Publication({"identifier": f"id_{i}", "publication_date": datetime(2023, 1, 1)})
            for i in range(10)
        ]
        atl = Atlas(pubs)

        # breakpoint()
        atl_proj = TestS2SBProjection.crt.project(atl)

        assert all([hasattr(pub, "publication_date") for pub in  atl.publications.values()])
        assert atl_proj.projection is None # was filtered

    def test_dummy_projection(self):
        pubs = [
            Publication(
                {
                    "identifier": f"id_{i}", 
                    "abstract": "blah blah blah",
                    "publication_date": datetime(2023, 1, 1)
                }
            )
            for i in range(10)
        ]
        atl = Atlas(pubs)

        atl_proj = TestS2SBProjection.crt.project(atl)

        projection = atl_proj.projection

        vector0 = projection.identifier_to_embedding("id_0")
        vector1 = projection.identifier_to_embedding("id_9")
        assert np.array_equal(vector0, vector1)

    def test_single_projection(self, tmp_path):
        path = tmp_path / atlas_dir
        path.mkdir()
        # Construct Atlas
        atl = TestS2SBProjection.crt.bibtex_to_atlas(single_pub_bibtex_fp)

        atl_proj = TestS2SBProjection.crt.project(atl)
        projection = atl_proj.projection

        identifier = list(atl.publications.keys())[0]
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
        atl_filtered = TestS2SBProjection.crt.filter(
            atl_exp_double, attributes=["abstract"]
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
        assert len(atl_exp_double) == 348

        atl_exp_double.save(path)
