"""The general container for data for any scientific publication, regardless of the API that was used to obtain it."""

# from datetime import datetime
from datetime import date, datetime
from typing import Any

from .misc.utils import standardize_month

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
        return [self.__getattribute__(field) if hasattr(self, field) else None for field in FIELDS + ADDITIONAL_FIELDS]

    @classmethod
    def from_csv_entry(cls, csv_entry: list):
        data = {k: v for k, v in dict(zip(FIELDS + ADDITIONAL_FIELDS, csv_entry)).items() if v == v } # check for nans

        if "publication_date" in data:
        # need to recast as datetime
            data["publication_date"] = datetime.strptime(
                data["publication_date"], "%Y-%m-%d",
            ).date()

        return cls(data)

    @classmethod
    def from_bibtex_entry(cls, bibtex_entry: dict):
        """Construct a Publication from a bibtex entry."""

        # identifier
        identifier = None
        if "identifier" in bibtex_entry:
            identifier = bibtex_entry["identifier"]
        elif "doi" in bibtex_entry:
            identifier = bibtex_entry["doi"]

        # abstract
        abstract = None
        if "abstract" in bibtex_entry:
            abstract = bibtex_entry["abstract"]

        # publication_date
        publication_date = None
        year = None
        month = None
        day = None
        if "year" in bibtex_entry:
            year = int(bibtex_entry["year"])
        if "month" in bibtex_entry:
            month_str = standardize_month(bibtex_entry["month"])
            month = datetime.strptime(month_str, "%B").month
        if "day" in bibtex_entry:
            day = int(bibtex_entry["day"])
        publication_date = date(
            year=year if year is not None else 1900,
            month=month if month is not None else 1,
            day=day if day is not None else 1,
        )
        # reset to None if default year
        if publication_date.year == 1900:
            publication_date = None
        
        # citation_count
        citation_count = None
        if "citation_count" in bibtex_entry: # it probably won't be
            citation_count = bibtex_entry["citation_count"]
        
        # Merge data
        primary_data = {
            "identifier": identifier,
            "publication_date": publication_date,
            "abstract": abstract,
            "citation_count": citation_count,
        }
        additional_data = {
            key: bibtex_entry[key] for key in ADDITIONAL_FIELDS if key in bibtex_entry
        }
        data = {k: v for k, v in (primary_data | additional_data).items() if v is not None}

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

        if "references" in data:
            val = data["references"]
            if not isinstance(val, list):
                raise ValueError
            
            self._references = val
        
        if "citation_count" in data:
            val = data["citation_count"]
            if not isinstance(val, int):
                breakpoint()
                raise ValueError

            self._citation_count = val

        ######################################################################
        # Other attributes
        ######################################################################  

        # data_copy = dict(data)
        # for key in FIELDS:
            # if key in data_copy:
        #         del data_copy[key]
        data_copy = {k:v for k,v in data.items() if k in ADDITIONAL_FIELDS}
        self.__dict__.update(data_copy)

        
            
        

