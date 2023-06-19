"""Test basic expansion functionality with Cartographer, which has some convenient wrappers around each API librarian."""

import bibtexparser

from sciterra.atlas import Atlas
from sciterra.cartography import Cartographer
from sciterra.librarians.semanticscholar import SemanticScholarLibrarian
from sciterra.publication import Publication
from sciterra.vectorization.scibert import SciBERTVectorizer

from .test_atlas import single_pub_bibtex_fp, atlas_single_dir

class TestCartographyS2SciBERT:

    librarian = SemanticScholarLibrarian()
    vectorizer = SciBERTVectorizer()
    crt = Cartographer(librarian, vectorizer)

    def test_empty_projection(self):
        
        pubs = [Publication({"identifier": f"id_{i}"}) for i in range(10)]
        atl = Atlas(pubs)

        projection = TestCartographyS2SciBERT.crt.project(atl)

        assert "identifiers_to_indices" in projection
        assert "indices_to_identifiers" in projection
        assert "embeddings" in projection

    def test_dummy_projection(self):
        
        pubs = [Publication(
            {"identifier": f"id_{i}", 
             "abstract": "blah blah blah"
             }) for i in range(10)]
        atl = Atlas(pubs)

        projection = TestCartographyS2SciBERT.crt.project(atl)

        assert "identifiers_to_indices" in projection
        assert "indices_to_identifiers" in projection
        assert "embeddings" in projection

    def test_single_projection(self, tmp_path):

        # Load single file from bibtex
        # Load expected values
        bibtex_fp = single_pub_bibtex_fp
        with open(bibtex_fp, "r") as f:
            bib_database = bibtexparser.load(f)
        entry: dict = bib_database.entries[0]

        path = tmp_path / atlas_single_dir
        path.mkdir()
        # Construct Atlas
        atl = Atlas.from_bibtex(single_pub_bibtex_fp)

        projection = TestCartographyS2SciBERT.crt.project(atl)

        identifier = atl.publications[0].identifier
        assert projection["identifiers_to_indices"] == {identifier:0}
        assert projection["indices_to_identifiers"] == (identifier,)
        assert projection["embeddings"].shape == (1,768) # (num_pubs, embedding_dim)


    def test_expand_single(self, tmp_path):

        # Load single file from bibtex
        # Load expected values
        bibtex_fp = single_pub_bibtex_fp
        with open(bibtex_fp, "r") as f:
            bib_database = bibtexparser.load(f)
        entry: dict = bib_database.entries[0]

        path = tmp_path / atlas_single_dir
        path.mkdir()
        # Construct Atlas
        atl = Atlas.from_bibtex(single_pub_bibtex_fp)

        atl_exp = TestCartographyS2SciBERT.crt.expand(atl)
