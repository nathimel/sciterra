"""Test basic expansion functionality with Cartographer, which has some convenient wrappers around each API librarian."""

from sciterra.cartography import expand
from sciterra.atlas import Atlas
from sciterra.librarians.semanticscholar import SemanticScholarLibrarian


class TestCartographyS2:

    librarian = SemanticScholarLibrarian()

    def test_expand(self):

        atl = Atlas([])

        atl_expanded = expand(atl)
