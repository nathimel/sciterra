"""The general container for data for any scientific publication, regardless of the API that was used to obtain it."""

# from datetime import datetime
from datetime import date
from typing import Any

# keys for data
FIELDS = [
    "identifier",
    "abstract",
    "publication_date",
    "citation_count",
    "url",
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
    def citation_count(self) -> int:
        return self._citation_count

    def to_csv_entry(self) -> list:
        """Convert publication to a list of values corresponding to FIELDS, for saving with other publications to a csv file."""
        return [self.__getattribute__(field) if hasattr(self, field) else None for field in FIELDS]

    @classmethod
    def from_csv_entry(cls, csv_entry: list):
        return cls(
            data={k: v for k, v in dict(zip(FIELDS, csv_entry)).items() if v == v } # check for nans
            )

    @classmethod
    def from_bibtex_entry(cls, bibtex_entry: dict):
        raise NotImplementedError

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
        
        if "citation_count" in data:
            val = data["citation_count"]
            if not isinstance(val, int):
                raise ValueError

            self._citation_count = val

        ######################################################################
        # Other attributes
        ######################################################################  

        if "url" in data:
            val = data["url"]
            if not isinstance(val, str):
                raise ValueError

            self.url = val
        

        data_copy = dict(data)
        for key in FIELDS:
            if key in data_copy:
                del data_copy[key]
        
        self.__dict__.update(data_copy)

        
            
        

