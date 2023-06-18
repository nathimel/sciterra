"""Test basic expansion functionality with Cartographer, which has some convenient wrappers around each API librarian."""

from sciterra.cartography import Cartographer
from sciterra.atlas import Atlas
from sciterra.librarians.semanticscholar import SemanticScholarLibrarian


class TestCartographerS2:

    librarian = SemanticScholarLibrarian()
    cartographer = Cartographer(librarian)

    def test_expand(self):

        atl = Atlas([])

        atl_expanded = TestCartographerS2.cartographer.expand(atl)
