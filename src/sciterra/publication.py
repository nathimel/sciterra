"""The general container for data for any scientific publication, regardless of the API that was used to obtain it."""

import warnings
from ast import literal_eval
from datetime import date, datetime
from typing import Any

from .misc.utils import standardize_month

"""Things a publication must have.

1. identifier
2. abstract
3. references -- a list of publication identifiers
4. citations -- a list of publication identifiers
5. publication date
6. citation count

"""


# keys for data
FIELDS = [
    "identifier",
    "abstract",
    "publication_date",
    "citation_count",
    "citations",
    "references",
]

ADDITIONAL_FIELDS = [
    "doi",
    "url",
    "title",
    "issn",
]


class Publication:
    """The Publication is a standardized container a scientific publication's retrieved data.
    
    Attributes:

        identifier:
            The string id that uniquely identifies the publication, used for    
                - storing in an Atlas
                - querying an API

        abstract:
            The string corresponding to the publication's abstract

        publication_date:
            A datetime representing the date of publication
        
        citation_count: 
            An int corresponding to the number of citations received by the publication
    """

    def __init__(self, data: dict = {}) -> None:
        # Below are the attributes we expect every publication to have. If a publication is missing these, it will be removed from analysis.
        self._identifier = None
        self._abstract = None
        self._publication_date = None
        self._citation_count = None

        # Regularize and store data, including but not limited to above attrs.
        self.init_attributes(data)

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def abstract(self) -> str:
        return self._abstract
    
    @property
    def publication_date(self) -> date:
        return self._publication_date
    
    @property
    def citations(self) -> list[str]:
        return self._citations
        
    @property
    def references(self) -> list[str]:
        return self._references
    
    @property
    def citation_count(self) -> int:
        return self._citation_count

    def to_csv_entry(self) -> list:
        """Convert publication to a list of values corresponding to FIELDS, for saving with other publications to a csv file."""
        # NOTE: for references and citations, you'll probably only be able to save the string identifiers. This means we should never expect references and citations to be ever more than lists of strings.
        return [self.__getattribute__(field) if hasattr(self, field) else None for field in FIELDS + ADDITIONAL_FIELDS]

    @classmethod
    def from_csv_entry(cls, csv_entry: list):
        data = {k: v for k, v in dict(zip(FIELDS + ADDITIONAL_FIELDS, csv_entry)).items() if v == v } # check for nans

        # Parse strings into their appropriate objects / literals
        if "citations" in data:
            data["citations"] = literal_eval(data["citations"])
        if "references in data":
            data["references"] = literal_eval(data["references"])

        if "publication_date" in data:
        # need to recast as datetime
            data["publication_date"] = datetime.strptime(
                data["publication_date"], "%Y-%m-%d",
            ).date()

        return cls(data)


    def __repr__(self) -> str:
        return "sciterra.publication.Publication:{}".format( self.identifier )

    def __str__( self ) -> str:
        return self.identifier
    
    def __hash__(self) -> int:
        return hash(self.__dict__.values())

    def __eq__(self, __value: object) -> bool:
        return self.__dict__ == __value.__dict__
    
    def __lt__(self, __value: object) -> bool:
        return str(self) < str(__value)
    
    def init_attributes(self, data) -> None:

        if "identifier" in data:
            val = data["identifier"]
            if not isinstance(val, str):
                raise ValueError
            self._identifier = val
        
        if "abstract" in data:
            val = data["abstract"]
            if not isinstance(val, str):
                raise ValueError
            self._abstract = val
        
        if "publication_date" in data:
            val = data["publication_date"]
            if not isinstance(val, date):
                raise ValueError
            self._publication_date = val

        if "citations" in data:
            val = data["citations"]
            if not isinstance(val, list):
                raise ValueError
            self._citations = val
        else:
            self._citations = []

        if "references" in data:
            val = data["references"]
            if not isinstance(val, list):
                raise ValueError
            self._references = val
        else:
            self._references = []
        
        if "citation_count" in data:
            val = data["citation_count"]
            if not isinstance(val, int):
                raise ValueError
            if len(self.citations) != val:
                # check that self.citations really does something
                warnings.warn(f"The length of the citations list ({len(self.citations)}) is different from citation_count ({val}). This is unexpected. Setting citation_count = len(self.citations). ")

            self._citation_count = val
        else:
            # we can use citations, but this is unexpected, so raise a warning.
            if self.citations:
                warnings.warn("Found an entry for 'citations' but no entry for citation_count; this is unexpected. Inferring value from citation_count.")
                self._citation_count = len(self.citations)

        ######################################################################
        # Other attributes
        ######################################################################  

        # data_copy = dict(data)
        # for key in FIELDS:
            # if key in data_copy:
        #         del data_copy[key]
        data_copy = {k:v for k,v in data.items() if k in ADDITIONAL_FIELDS}
        self.__dict__.update(data_copy)

        
            
        
