from .librarian import Librarian
from .adslibrarian import ADSLibrarian
from .s2librarian import SemanticScholarLibrarian

"""Why is there not an ArxivLibrarian? For now, we are restricting to APIs that allow us to traverse literature graphs, and arxiv does not have one. While there is a useful pip-installable package for querying the arxiv api for papers, https://pypi.org/project/arxiv/, the returned object does not have information on references and citations. However, it may still be possible to obtain a large sample of publications with abstracts and submission dates (though no citation counts), because the arxiv API's limit for a single query is 300,000 results.
"""
