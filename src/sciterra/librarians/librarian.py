from ..publication import Publication
from typing import Any

class Librarian:

    def __init__(self) -> None:
        pass

    def query_publications(
        self, 
        identifiers: list[str], 
        *args, 
        call_size: int = None,
        n_attempts_per_query: int = None,
        **kwargs,
        ) -> list[Publication]:
        """Call an API and retrieve the publications corresponding to str identifiers.
        
        Args:
            n_attempts_per_query: Number of attempts to access the API per query. Useful when experiencing connection issues.

            call_size: (int): maximum number of papers to call API for in one query; if less than `len(paper_ids)`, chunking will be performed.        
        """
        raise NotImplementedError

    def convert_publication(self, pub: Any, *args, **kwargs):
        """Convert an API-specific resulting publication data structure into a sciterra Publication object."""
        raise NotImplementedError
    
    def convert_publications(self, pubs: list[Any], *args, **kwargs):
        """Convert a list of API-specific publication data structures into a list of sciterra Publications."""

        