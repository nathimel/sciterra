from ..publication import Publication
from datetime import date

from typing import Any
from multiprocessing import Pool
from tqdm import tqdm


class Librarian:
    def __init__(self) -> None:
        pass

    def bibtex_entry_identifier(self, bibtex_entry: dict) -> str:
        """Parse a bibtex entry for a usable unique identifier appropriate to the API."""
        raise NotImplementedError

    def get_publications(
        self,
        identifiers: list[str],
        *args,
        call_size: int = None,
        n_attempts_per_query: int = None,
        convert: bool = True,
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

    def convert_publications(
        self,
        papers: list,
        *args,
        multiprocess: bool = True,
        num_processes=6,
        **kwargs,
    ) -> list[Publication]:
        """Convet a list of API-specific results to sciterra Publications, possibly using multiprocessing."""

        if not multiprocess:
            return [
                self.convert_publication(
                    paper,
                )
                for paper in papers
            ]

        with Pool(processes=num_processes) as p:
            publications = list(
                tqdm(
                    p.imap(self.convert_publication, papers),
                    total=len(papers),
                )
            )

        return publications
