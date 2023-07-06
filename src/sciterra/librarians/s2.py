from typing import Any
from tqdm import tqdm

from sciterra.publication import Publication

from ..publication import Publication
from .librarian import Librarian
from ..misc.utils import chunk_ids, keep_trying

from semanticscholar import SemanticScholar
from semanticscholar.Paper import Paper

from requests.exceptions import ReadTimeout, ConnectionError
from semanticscholar.SemanticScholarException import ObjectNotFoundExeception

from multiprocessing import Pool

##############################################################################
# Constants
##############################################################################

# NOTE: semantic scholar will truncate total number of references, citations each at 10,000 for the entire batch.
QUERY_FIELDS = [
    'abstract',
    'externalIds', # supports ArXiv, MAG, ACL, PubMed, Medline, PubMedCentral, DBLP, DOI
    'url', # as a possible external id
    'citations.externalIds',
    'citations.url',
    'references.externalIds',
    'references.url',
    'citationStyles', # supports a very basic bibtex that we will augment
    'publicationDate', # if available, type datetime.datetime (YYYY-MM-DD)
]

# The following types of IDs are supported
EXTERNAL_IDS  = [
    'DOI', 
    'ArXiv', 
    'CorpusId',
    'MAG', 
    'ACL', 
    'PubMed', 
    'Medline', 
    'PubMedCentral', 
    'DBLP',
    'URL',
    ]

# for storing the results from above, we avoid dot operator to avoid attribute error, but note that everything above will be included.
STORE_FIELDS = [
    'abstract',
    'externalIds', 
    'url', 
    'citations',
    'references',
    'citationStyles', 
    'publicationDate', 
]

# Attributes to save via save_data
ATTRS_TO_SAVE = [
    'paper', 
    'abstract',
    'citations',
    'references',
    'bibcode',
    'entry_date',
    'notes',
    'unofficial_flag',
    'citation',
    'stemmed_content_words',
]

ALLOWED_EXCEPTIONS = (
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

    def get_publications(
        self, 
        identifiers: list[str], 
        *args, 
        call_size: int = CALL_SIZE,        
        n_attempts_per_query: int = NUM_ATTEMPTS_PER_QUERY,
        convert: bool = True,
        **kwargs,
        ) -> list[Publication]:
        """Use the (unofficial) S2 python package, which calls the Semantic Scholar API to retrieve publications from the S2AG.

        Args:
            identifiers: the str ids required for querying. See EXTERNAL_IDS

            n_attempts_per_query: Number of attempts to access the API per query. Useful when experiencing connection issues.

            call_size: maximum number of papers to call API for in one query; if less than `len(paper_ids)`, chunking will be performed.

            convert: whether to convert resulting SemanticScholar Papers to sciterra Publications (True by default).

        Returns:
            the list of publications (or Papers)
        """

        identifiers = list( identifiers )

        if not identifiers:
            return []

        total = len(identifiers)
        chunked_ids = chunk_ids(identifiers, call_size = call_size)

        if None in identifiers:
            # any Nones should have been handled by this point
            raise Exception("Passed `identifiers` contains None.")

        print( f'Querying Semantic Scholar for {len(identifiers)} total papers.')
        papers = []

        pbar = tqdm(desc=f'progress using call_size={call_size}', total=total)
        for ids in chunked_ids:
            
            @keep_trying( n_attempts=n_attempts_per_query, )
            def get_papers() -> list[Paper]:
                if call_size > 1:
                    result = self.sch.get_papers(
                        paper_ids=ids,
                        fields=QUERY_FIELDS,
                    )
                else:
                    # typically completes about 100 queries per minute.
                    result = [
                        self.sch.get_paper(
                        paper_id=paper_id, 
                        fields=QUERY_FIELDS,
                        ) for paper_id in ids
                    ]
                return result

            papers.extend(get_papers())
            pbar.update(len(ids))
        pbar.close()

        if not convert:
            return papers
        return self.convert_publications(papers, identifiers=identifiers)

    def convert_publication(self, paper: Paper, *args, **kwargs) -> Publication:
        """Convert a SemanticScholar Paper object to a sciterra.publication.Publication. """

        # to be consistent with identifiers (e.g., to avoid storing the same publication twice), we use the identifier passed to the query. Most of the time this will just be the paperId.
        identifier = paper.paperId
        if "identifier" in kwargs:
            identifier = kwargs["identifier"]

        # get doi from externalids
        doi = None
        if "DOI" in paper.externalIds:
            doi = paper.externalIds["DOI"]

        # parse data
        data = {
            # primary fields
            "identifier": identifier,
            "abstract": paper.abstract,
            "publication_date": paper.publicationDate,
            "citations": paper.citations,
            "references": paper.references,
            "citation_count": paper.citationCount,
            # additional fields
            "doi": doi,
            "url": paper.title,
            "title": paper.title if hasattr(paper, "title") else None,
            "issn": paper.issn if hasattr(paper, "issn") else None,
        }
        data = {k:v for k,v in data.items() if v is not None}

        return Publication(data)
    
    def convert_publications(
        self, 
        papers: list[Paper], 
        *args, 
        multiprocess: bool = True,
        num_processes = 6, 
        **kwargs,
        ) -> list[Publication]:
        """Convet a list of SemanticScholar Papes to sciterra Publications, possibly using multiprocessing."""

        identifiers = []
        if "identifiers" in kwargs:
            identifiers = kwargs["identifiers"]

        if not multiprocess:
            return [self.convert_publication(
                paper, 
                **{"identifier": identifiers[idx]} if identifiers else {},
                ) for idx, paper in enumerate(papers)]


        with Pool(num_processes) as p:
            async_results = [
                p.apply_async(
                self.convert_publication,
                args=[paper],
                kwds={"identifier": identifiers[idx]} if identifiers else {},
                )
                for idx, paper in enumerate(papers)
            ]
            p.close()
            p.join()

        return [async_result.get() for async_result in tqdm(async_results)]
