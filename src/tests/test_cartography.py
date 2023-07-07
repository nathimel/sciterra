"""Test basic expansion functionality with Cartographer, which has some convenient wrappers around each API librarian."""

import bibtexparser

from sciterra.atlas import Atlas
from sciterra.cartography import Cartographer
from sciterra.librarians.s2librarian import SemanticScholarLibrarian
from sciterra.publication import Publication
from sciterra.vectorization.scibert import SciBERTVectorizer

from .test_atlas import single_pub_bibtex_fp, ten_pub_bibtex_fp, realistic_bibtex_fp

##############################################################################
# SemanticScholar x SciBERT
##############################################################################

atlas_dir = "atlas_tmpdir"

# NOTE: The purpose of mapping a 
class TestS2BibtexToAtlas:

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

        pub, = list(atl.publications.values())

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
        assert len(atl) == 28


class TestS2SBProjection:

    librarian = SemanticScholarLibrarian()
    vectorizer = SciBERTVectorizer()
    crt = Cartographer(librarian, vectorizer)

    def test_empty_projection(self):
        
        pubs = [Publication({"identifier": f"id_{i}"}) for i in range(10)]
        atl = Atlas(pubs)

        projection = TestS2SBProjection.crt.project(atl)

        assert "identifiers_to_indices" in projection
        assert "indices_to_identifiers" in projection
        assert "embeddings" in projection

    def test_dummy_projection(self):
        
        pubs = [Publication(
            {"identifier": f"id_{i}", 
             "abstract": "blah blah blah"
             }) for i in range(10)]
        atl = Atlas(pubs)

        projection = TestS2SBProjection.crt.project(atl)

        assert "identifiers_to_indices" in projection
        assert "indices_to_identifiers" in projection
        assert "embeddings" in projection

    def test_single_projection(self, tmp_path):

        path = tmp_path / atlas_dir
        path.mkdir()
        # Construct Atlas
        atl = TestS2SBProjection.crt.bibtex_to_atlas(single_pub_bibtex_fp)

        projection = TestS2SBProjection.crt.project(atl)

        identifier = list(atl.publications.keys())[0]
        assert projection["identifiers_to_indices"] == {identifier:0}
        assert projection["indices_to_identifiers"] == (identifier,)
        assert projection["embeddings"].shape == (1,768) # (num_pubs, embedding_dim)


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
        entry: dict = bib_database.entries[0]

        path = tmp_path / atlas_dir
        path.mkdir()
        # Construct Atlas
        atl = TestS2SBProjection.crt.bibtex_to_atlas(bibtex_fp)

        pub = list(atl.publications.values())[0]
        ids = pub.citations + pub.references

        atl_exp = TestS2SBProjection.crt.expand(atl)

        assert len(atl_exp) > len(atl)
        # so far this holds, but things that aren't our fault could make it fail.
        assert len(atl_exp) == len(ids)

