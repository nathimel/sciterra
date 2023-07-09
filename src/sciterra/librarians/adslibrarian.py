import ads

from sciterra.publication import Publication
from ..publication import Publication
from .librarian import Librarian

from ..misc.utils import keep_trying


QUERY_FIELDS = [
    "bibcode",
    "abstract",
    "title",
    "entry_date", # datetime (earliest possible)    
    "pubdate", # a datetime
    "year", # int
    "citation_count",
    "citation", # list
    "reference", # list
]

ALLOWED_EXCEPTIONS = (
    ads.exceptions.APIResponseError,
)

EXTERNAL_IDS = [
    "DOI", # returns a list
    "arXiv", # returns a str
    "bibcode" # returns a str, preferred
]

class ADSLibrarian(Librarian):
    def __init__(self) -> None:
        super().__init__()


    def bibtex_entry_identifier(self, bibtex_entry: dict) -> str:
        """Parse a bibtex entry for a usable identifier for querying ADS (see EXTERNAL_IDS)."""
        pass


    def get_publications(self, identifiers: list[str], *args, call_size: int = None, n_attempts_per_query: int = None, **kwargs) -> list[Publication]:
        """ADS baby
        """

        # TODO: loop over identifiers and get results
        query = None


        @keep_trying(
            n_attempts=n_attempts_per_query,
            allowed_exceptions=ALLOWED_EXCEPTIONS,
            sleep_after_attempt=2,
        )
        def ads_query():
            ads_query = ads.SearchQuery(
                query_dict = {
                    "q": query,
                    "fl": QUERY_FIELDS,
                }
            )
            article, = list(ads_query) # retrieve from generator
            return article


