"""The general container for data for any scientific publication, regardless of the API that was used to obtain it."""

from datetime import datetime


class Publication:
    """The Publication is a standardized container a scientific publication's retrieved data.
    
    Attributes:

        identifier:
            The string id that uniquely identifies the publication, used for    
                - storing in an Atlas
                - querying an API
    """

    def __init__(self, data: dict) -> None:
        # Below are the attributes we expect every publication to have. If a publication is missing these, it will be removed from analysis.
        self._identifier = None
        self._abstract = None
        self._publication_date = None
        self._citation_count = None

        # Regularize and store data, including but not limited to above attrs.
        self.init_attributes(data)

    @property
    def abstract(self) -> str:
        return self._abstract

    @property
    def abstract(self) -> str:
        return self._abstract
    
    @property
    def publication_date(self) -> datetime:
        return self._publication_date
    
    @property
    def citation_count(self) -> int:
        return self._citation_count


    @classmethod
    def from_bibtex_entry(cls, bibtex_entry: dict):
        return cls(None)

    def __repr__(self) -> str:
        return "sciterra.publication.Publication:{}".format( self.identifier )

    def __str__( self ) -> str:
        return self.identifier
    
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
            if not isinstance(val, datetime):
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
        
            
        

