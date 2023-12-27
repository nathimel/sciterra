import ads

from ads.search import Article
from datetime import date, datetime

from sciterra.mapping.publication import Publication
from ..mapping.publication import Publication
from .librarian import Librarian

from ..misc.utils import chunk_ids, keep_trying, get_verbose

from tqdm import tqdm

import warnings

CALL_SIZE = (
    50  # handles more than 2000, much better than S2; but easy to hit TooManyRequests
)
NUM_ATTEMPTS_PER_QUERY = 10

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
    "identifier",  # list of external ids
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
        call_size: int = CALL_SIZE,
        n_attempts_per_query: int = NUM_ATTEMPTS_PER_QUERY,
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

        if call_size is None:
            call_size = CALL_SIZE

        total = len(bibcodes)
        chunked_ids = chunk_ids(
            bibcodes,
            call_size=call_size,
        )

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
                        0  # screw black, this is ugly
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

        verbose = get_verbose(kwargs)

        # to be consistent with identifiers (e.g., to avoid storing the same publication twice), we always use the bibcode.
        identifier = article.bibcode

        def process_date(date_str: str) -> str:
            # sometimes there is extra data
            date_str = date_str[:10]  # e.g. yyyy-mm-dd
            # sometimes ads has 00 for month or day
            if date_str[-2:] == "00":
                date_str[-2:] = "01"
            if date_str[-5:-3] == "00":
                date_str[-5:-3] = "01"
            date_ = datetime.strptime(date_str, "%Y-%m-%d")
            return date_

        # Parse date from datetime or year
        if hasattr(article, "entry_date"):
            publication_date = process_date(article.entry_date)
        elif hasattr(article, "pubdate"):
            publication_date = process_date(article.pubdate)
        elif hasattr(article, "year"):
            publication_date = date(article.year, 1, 1)
        else:
            publication_date = None

        # get doi from Article identifiers
        # warning: ADS tracks two DOIs: for official and arxiv
        doi = None
        if hasattr(article, "identifier"):
            dois = [item for item in article.identifier if item[:3] == "10."]
            if dois:
                doi = dois[0]

        # Process citation data
        citations = article.citation
        references = article.reference

        citation_count = article.citation_count
        if (
            (citation_count is not None)
            and (citations is not None)
            and (citation_count != len(citations))
            and verbose
        ):
            warnings.warn(
                f"The length of the citations list ({len(citations)}) is different from citation_count ({citation_count})"
            )
            if "infer_citation_count" in kwargs and kwargs["infer_citation_count"]:
                if verbose:
                    warnings.warn("Setting citation_count = {len(citations)}.")
                citation_count = len(citations)

        # N.B.: we could manually add fields_of_study=['Physics'], but unclear that would be informative. Ads has the field 'arxiv_class', but that's probably more fine-grained than we need.
        data = {
            # primary fields
            "identifier": identifier,
            "abstract": article.abstract,
            "publication_date": publication_date,
            "citations": citations,
            "references": references,
            "citation_count": citation_count,
            # additional fields
            "doi": doi,
            "title": article.title,
        }
        data = {k: v for k, v in data.items() if v is not None}

        return Publication(data)
