"""Test basic expansion functionality with Cartographer, which has some convenient wrappers around each API librarian."""

from sciterra.atlas import Atlas
from sciterra.cartography import Cartographer
from sciterra.librarians.semanticscholar import SemanticScholarLibrarian
from sciterra.publication import Publication
from sciterra.vectorization.scibert import SciBERTVectorizer


class TestCartographyS2:

    librarian = SemanticScholarLibrarian()
    vectorizer = SciBERTVectorizer()
    crt = Cartographer(librarian, vectorizer)

    def test_empty_projection(self):
        
        pubs = [Publication({"identifier": f"id_{i}"}) for i in range(10)]
        atl = Atlas(pubs)

        projection = TestCartographyS2.crt.project(atl)

        assert "identifiers_to_indices" in projection
        assert "indices_to_identifiers" in projection
        assert "embeddings" in projection

    def test_nonempty_projection(self):
        
        pubs = [Publication(
            {"identifier": f"id_{i}", 
             "abstract": "blah blah blah"
             }) for i in range(10)]
        atl = Atlas(pubs)

        projection = TestCartographyS2.crt.project(atl)

        assert "identifiers_to_indices" in projection
        assert "indices_to_identifiers" in projection
        assert "embeddings" in projection