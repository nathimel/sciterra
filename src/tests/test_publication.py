"""Test the basic publication wrapper."""

import bibtexparser
import pytest

from datetime import datetime
from sciterra.mapping.publication import Publication


single_pub_bibtex_fp = "src/tests/data/single_publication.bib"


class TestPublication:
    def test_empty_publication(self):
        # Publication must have data
        with pytest.raises(TypeError) as e_info:
            pub = Publication()

    def test_dummy_publication(self):
        data = {
            "identifier": "exampleidentifierstring",
            "abstract": "Example abstract text.",
            "publication_date": datetime.today().date(),
            "citation_count": 0,
            "url": "exampleurl.com",
            "extra": Publication,  # extra garbage
        }
        pub = Publication(data)

        assert pub.identifier == "exampleidentifierstring"
        assert str(pub) == pub.identifier
        assert pub.abstract == "Example abstract text."
        assert pub.publication_date == datetime.today().date()
        assert pub.citation_count == 0
        assert pub.url == "exampleurl.com"
        assert not hasattr(pub, "extra")
