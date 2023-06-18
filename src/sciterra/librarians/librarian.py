from publication import Publication

class Librarian:

    def __init__(self) -> None:
        pass

    def query_publications(self, identifiers: list[str], *args, **kwargs) -> list[Publication]:
        """Call an API and retrieve the publications corresponding to str identifiers."""
        raise NotImplementedError