import ads

from ads.search import Article
from datetime import date

from sciterra.publication import Publication
from ..publication import Publication
from .librarian import Librarian

from ..misc.utils import chunk_ids, keep_trying

from tqdm import tqdm


QUERY_FIELDS = [
    "bibcode",  # str
    "abstract",  # str
    "title",  # list
    "entry_date",  # datetime (earliest possible)
    "pubdate",  # a datetime
    "year",  # int
    "citation_count",
    "citation",  # list
    "reference",  # list
]

ALLOWED_EXCEPTIONS = (ads.exceptions.APIResponseError,)

EXTERNAL_IDS = [
    "DOI",  # returns a list
    "arXiv",  # returns a str
    "bibcode",  # returns a str, preferred
]


class ADSLibrarian(Librarian):
    def __init__(self) -> None:
        super().__init__()

    def bibtex_entry_identifier(self, bibtex_entry: dict) -> str:
        """Parse a bibtex entry for a usable identifier for querying ADS (see EXTERNAL_IDS)."""
        identifier = None
        if "bibcode" in bibtex_entry:
            identifier = bibtex_entry["bibcode"]
        elif "doi" in bibtex_entry:
            identifier = f"doi:{bibtex_entry['doi']}"
        elif "arxiv" in bibtex_entry:
            identifier = f"arxiv:{bibtex_entry['arxiv']}"
        return identifier

    def get_publications(
        self,
        bibcodes: list[str],
        *args,
        call_size: int = None,
        n_attempts_per_query: int = None,
        convert: bool = True,
        **kwargs,
    ) -> list[Publication]:
        """Use the NASA ADS python package, which calls the ADS API to retrieve publications.

        Args:
            bibcodes: the str ids required for querying. While it is possible to use one of EXTERNAL_IDS to query, if ADS returns a paper at all, it will return a bibcode, so it is preferred to use bibcodes.

            n_attempts_per_query: Number of attempts to access the API per query. Useful when experiencing connection issues.

            call_size: maximum number of papers to call API for in one query; if less than `len(bibcodes)`, chunking will be performed.

            convert: whether to convert each resulting ADS Article to sciterra Publications (True by default).

        Returns:
            the list of publications (or Papers)
        """
        bibcodes = list(bibcodes)

        if not bibcodes:
            return []

        total = len(bibcodes)
        chunked_ids = chunk_ids(bibcodes, call_size=call_size)

        if None in bibcodes:
            # any Nones should have been handled by this point
            raise Exception("Passed `bibcodes` contains None.")

        print(f"Querying ADS for {len(bibcodes)} total papers.")
        papers = []
        pbar = tqdm(desc=f"progress using call_size={call_size}", total=total)
        for ids in chunked_ids:

            @keep_trying(
                n_attempts=n_attempts_per_query,
                allowed_exceptions=ALLOWED_EXCEPTIONS,
                sleep_after_attempt=2,
            )
            def get_papers() -> list[Article]:
                return [
                    list(
                        ads.SearchQuery(
                            query_dict={
                                "q": query,
                                "fl": QUERY_FIELDS,
                            }
                        )
                    )[
                        0
                    ]  # retrieve from generator
                    for query in ids
                ]

            papers.extend(get_papers())
            pbar.update(len(ids))

        pbar.close()

        if not convert:
            return papers
        return self.convert_publications(
            papers,
            *args,
            **kwargs,
        )

    def convert_publication(self, article: Article, *args, **kwargs) -> Publication:
        """Convert a ADS Article object to a sciterra.publication.Publication."""
        if article is None:
            return

        # to be consistent with identifiers (e.g., to avoid storing the same publication twice), we always use the bibcode.
        identifier = article.bibcode

        # datetime.strptime(
        # data['publicationDate'], '%Y-%m-%d')

        # Parse date from datetime or year
        if hasattr(article, "entry_date"):
            publication_date = article.entry_date
        elif hasattr(article, "pubdate"):
            publication_date = article.pubdate
        elif hasattr(article, "year"):
            publication_date = date(article.year, 1, 1)
        else:
            publication_date = None
