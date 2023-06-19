
from ..publication import Publication
from .librarian import Librarian

class SemanticScholarLibrarian(Librarian):

    def __init__(self) -> None:
        super().__init__()

    def query_publications(self, identifiers: list[str], *args, **kwargs) -> list[Publication]:
        """Use the (unofficial) S2 python package, which calls the Semantic Scholar API to retrieve publications from the S2AG.
        """
        raise NotImplementedError
