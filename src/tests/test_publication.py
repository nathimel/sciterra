"""Test the basic publication wrapper."""

from datetime import datetime
from sciterra.publication import Publication

import bibtexparser

single_pub_bibtex_fp = "src/tests/data/single_publication.bib"

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
        assert not hasattr(pub, "extra")

    def test_from_bibtex(self):
        bibtex_fp = single_pub_bibtex_fp
        with open(bibtex_fp, "r") as f:
            bib_database = bibtexparser.load(f)

        entry: dict = bib_database.entries[0]

        pub = Publication.from_bibtex_entry(entry)

        # main attributes
        assert pub.identifier in entry.values()
        assert pub.abstract == entry["abstract"]
        assert pub.publication_date.year == int(entry["year"])

        # other attributes
        assert pub.url == entry["url"]
        assert pub.issn == entry["issn"]
        assert pub.doi == entry["doi"]
        assert pub.title == entry["title"]
