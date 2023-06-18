"""Test the basic publication wrapper."""

from datetime import datetime
from sciterra.publication import Publication

class TestPublication:

    def test_empty_publication(self):
        
        pub = Publication()
        assert pub.identifier is None

    def test_dummy_publication(self):

        data = {
            "identifier": "exampleidentifierstring",
            "abstract": "Example abstract text.",
            "publication_date": datetime.today().date(),
            "citation_count": 0,
            "url": "exampleurl.com",
            "extra": Publication, # extra garbage
        }
        pub = Publication(data)

        assert pub.identifier == "exampleidentifierstring"
        assert pub.abstract == "Example abstract text."
        assert pub.publication_date == datetime.today().date()
        assert pub.citation_count == 0
        assert pub.url == "exampleurl.com"
        assert pub.extra == Publication
        breakpoint()

    
    