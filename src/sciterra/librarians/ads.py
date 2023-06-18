
from publication import Publication
from librarian import Librarian

class ADSLibrarian(Librarian):

    def __init__(self) -> None:
        super().__init__()

    def query_publications(self, identifiers: list[str], *args, **kwargs) -> list[Publication]:
        raise NotImplementedError