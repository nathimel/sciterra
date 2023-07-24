import warnings

from datetime import date

from typing import Any
from tqdm import tqdm


from sciterra.mapping.publication import Publication

from ..mapping.publication import Publication
from .librarian import Librarian
from ..misc.utils import chunk_ids, keep_trying, get_verbose

from semanticscholar import SemanticScholar
from semanticscholar.Paper import Paper

from requests.exceptions import ReadTimeout, ConnectionError
from semanticscholar.SemanticScholarException import ObjectNotFoundExeception

##############################################################################
# Constants
##############################################################################

# NOTE: semantic scholar will truncate total number of references, citations each at 10,000 for the entire batch.
QUERY_FIELDS = [
    "year",
    "abstract",
    "title",  # useful for inspection
    "externalIds",  # supports ArXiv, MAG, ACL, PubMed, Medline, PubMedCentral, DBLP, DOI
    "citationCount",
    "url",  # as a possible external id
    "citations.externalIds",
    "citations.url",
    "references.externalIds",
    "references.url",
    "citationStyles",  # supports a very basic bibtex that we will augment
    "publicationDate",  # if available, type datetime.datetime (YYYY-MM-DD)
]

# The following types of IDs are supported
EXTERNAL_IDS = [
    "DOI",
    "ArXiv",
    "CorpusId",
    "MAG",
    "ACL",
    "PubMed",
    "Medline",
    "PubMedCentral",
    "DBLP",
    "URL",
]

# for storing the results from above, we avoid dot operator to avoid attribute error, but note that everything above will be included.
# n.b.: no idea what i meant above here
STORE_FIELDS = [
    "abstract",
    "externalIds",
    "url",
    "citations",
    "references",
    "citationStyles",
    "publicationDate",
]

# Attributes to save via save_data
ATTRS_TO_SAVE = [
    "paper",
    "abstract",
    "citations",
    "references",
    "bibcode",
    "entry_date",
    "notes",
    "unofficial_flag",
    "citation",
    "stemmed_content_words",
]

ALLOWED_EXCEPTIONS = (
    Exception,  # "Internal Service Error"
    ReadTimeout,
    ConnectionError,
    ObjectNotFoundExeception,
)
CALL_SIZE = 10
NUM_ATTEMPTS_PER_QUERY = 50

##############################################################################
# Main librarian class
##############################################################################


class SemanticScholarLibrarian(Librarian):
    def __init__(self) -> None:
        self.sch = SemanticScholar()
        super().__init__()

    def bibtex_entry_identifier(self, bibtex_entry: dict) -> str:
        """Parse a bibtex entry for a usable identifier for querying SemanticScholar (see EXTERNAL_IDS)."""
        identifier = None
        if "paper_id" in bibtex_entry:
            identifier = bibtex_entry["paper_id"]
        elif "doi" in bibtex_entry:
            identifier = f"DOI:{bibtex_entry['doi']}"
        return identifier

    def get_publications(
        self,
        paper_ids: list[str],
        *args,
        call_size: int = CALL_SIZE,
        n_attempts_per_query: int = NUM_ATTEMPTS_PER_QUERY,
        convert: bool = True,
        **kwargs,
    ) -> list[Publication]:
        """Use the (unofficial) S2 python package, which calls the Semantic Scholar API to retrieve publications from the S2AG.

        Args:
            paper_ids: the str ids required for querying. While it is possible to use one of EXTERNAL_IDS to query, if SemanticScholar returns a paper at all, it will return a paperId, so it is preferred to use paperIds.

            n_attempts_per_query: Number of attempts to access the API per query. Useful when experiencing connection issues.

            call_size: maximum number of papers to call API for in one query; if less than `len(paper_ids)`, chunking will be performed. Maximum that S2 allows is 500.

            convert: whether to convert each resulting SemanticScholar Paper to sciterra Publications (True by default).

        Returns:
            the list of publications (or Papers)
        """
        paper_ids = list(paper_ids)

        if not paper_ids:
            return []

        total = len(paper_ids)
        chunked_ids = chunk_ids(paper_ids, call_size=call_size)

        if None in paper_ids:
            # any Nones should have been handled by this point
            raise Exception("Passed `paper_ids` contains None.")

        print(f"Querying Semantic Scholar for {len(paper_ids)} total papers.")
        papers = []
        pbar = tqdm(desc=f"progress using call_size={call_size}", total=total)
        for ids in chunked_ids:

            @keep_trying(
                n_attempts=n_attempts_per_query,
                allowed_exceptions=ALLOWED_EXCEPTIONS,
                sleep_after_attempt=2,
            )
            def get_papers() -> list[Paper]:
                if call_size > 1:
                    result = self.get_papers(
                        paper_ids=ids,
                        fields=QUERY_FIELDS,
                    )
                else:
                    # typically completes about 100 queries per minute.
                    result = [
                        self.get_paper(
                            paper_id=paper_id,
                            fields=QUERY_FIELDS,
                        )
                        for paper_id in ids
                    ]
                return result

            papers.extend(get_papers())
            pbar.update(len(ids))
        pbar.close()

        if not convert:
            return papers
        return self.convert_publications(  # may contain Nones!
            papers,
            *args,
            **kwargs,
        )

    def convert_publication(self, paper: Paper, *args, **kwargs) -> Publication:
        """Convert a SemanticScholar Paper object to a sciterra.publication.Publication."""
        if paper is None:
            return

        verbose = get_verbose(kwargs)

        # to be consistent with identifiers (e.g., to avoid storing the same publication twice), we always use the paperId.
        identifier = paper.paperId

        # Parse date from datetime or year
        if paper.publicationDate is not None:
            publication_date = paper.publicationDate.date()
        elif paper.year is not None:
            publication_date = date(paper.year, 1, 1)
        else:
            publication_date = None

        # get doi from externalids
        doi = None
        if "DOI" in paper.externalIds:
            doi = paper.externalIds["DOI"]

        # convert citations/references from lists of Papers to identifiers
        citations = [
            paper.paperId for paper in paper.citations if paper.paperId is not None
        ]  # no point using recursion assuming identifier=paperId
        references = [
            paper.paperId for paper in paper.references if paper.paperId is not None
        ]

        citation_count = paper.citationCount
        if citation_count != len(citations) and verbose:
            warnings.warn(
                f"The length of the citations list ({len(citations)}) is different from citation_count ({citation_count})"
            )
        if (
            "infer_citation_count" in kwargs
            and kwargs["infer_citation_count"]
            and verbose
        ):
            warnings.warn("Setting citation_count = {len(citations)}.")
            citation_count = len(citations)

        data = {
            # primary fields
            "identifier": identifier,
            "abstract": paper.abstract,
            "publication_date": publication_date,
            "citations": citations,
            "references": references,
            "citation_count": citation_count,
            # additional fields
            "doi": doi,
            "url": paper.url if hasattr(paper, "url") else None,
            "title": paper.title if hasattr(paper, "title") else None,
            "issn": paper.issn if hasattr(paper, "issn") else None,
        }
        data = {k: v for k, v in data.items() if v is not None}

        return Publication(data)

    # We write this minimally different function from SemanticScholar.get_papers so that others dont need to fork our version of semantic scholar.
    def get_papers(self, paper_ids: list[str], fields: list[str]):
        """Custom function for calling the S2 API that doesn't fail on empty results."""
        if not fields:
            fields = Paper.SEARCH_FIELDS

        url = f"{self.sch.api_url}/paper/batch"

        fields = ",".join(fields)
        parameters = f"&fields={fields}"

        payload = {"ids": paper_ids}

        data = self.sch._requester.get_data(
            url, parameters, self.sch.auth_header, payload
        )
        papers = [
            Paper(item) if item is not None else None for item in data
        ]  # added condition

        return papers

    # "
    def get_paper(self, paper_id: str, fields: list[str]):
        """Custom function for calling the S2 API that doesn't fail on empty results."""
        if not fields:
            fields = Paper.FIELDS

        url = f"{self.api_url}/paper/{paper_id}"

        fields = ",".join(fields)
        parameters = f"&fields={fields}"

        data = self.sch._requester.get_data(url, parameters, self.auth_header)
        paper = Paper(data) if data is not None else None  # added condition

        return paper
