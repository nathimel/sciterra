"""Test basic Atlas functionality, independently of API. To obtain realistic publication data, should probably read in a .bib file."""

import bibtexparser

import pandas as pd

from sciterra.atlas import Atlas
from sciterra.publication import Publication


atlas_empty_dir = "atlas_empty"
atlas_single_dir = "atlas_single"
atlas_10_dir = "atlas_10"
class TestAtlasDummy:

    def test_create_empty_atlas(self):

        atl = Atlas([])
        assert atl.publications == []

    def test_save_empty_atlas(self, tmp_path):

        path = tmp_path / atlas_empty_dir
        path.mkdir()
        atl = Atlas([])
        atl.save(path)

    def test_save_load_empty_atlas(self, tmp_path):

        path = tmp_path / atlas_empty_dir
        path.mkdir()
        embedding_path = path / "publications.csv"
        pd.DataFrame().to_csv(embedding_path)

        atl = Atlas.load(path)
        assert atl.publications == Atlas([]).publications

    """Test an Atlas with a single publication."""

    def test_atlas_single(self):
        pub = Publication({"identifier": "id"})
        atl = Atlas([pub])
        assert atl[str(pub)] == pub

    def test_atlas_single_duplicate(self):

        pub = Publication({"identifier":"id"})
        atl = Atlas([pub,pub])
        assert atl[str(pub)] == pub
        assert len(atl) == 1

    def test_save_atlas_single(self, tmp_path):

        path = tmp_path / atlas_single_dir
        path.mkdir()
        atl = Atlas([Publication({"identifier": "id"})])
        atl.save(path)

    def test_save_load_atlas_single(self, tmp_path):

        path = tmp_path / atlas_single_dir
        path.mkdir()
        atl = Atlas([Publication({"identifier": "id"})])
        atl.save(path)

        atl_loaded = Atlas.load(path)
        assert atl.publications == atl_loaded.publications

    """Test an Atlas with 10 publications."""

    def test_atlas_10(self):
        
        pubs = [Publication({"identifier": f"id_{i}"}) for i in range(10)]
        atl = Atlas(pubs)
        for pub in pubs:
            assert atl[str(pub)] == pub

    def test_save_atlas_10(self, tmp_path):

        path = tmp_path / atlas_10_dir
        path.mkdir()

        pubs = [Publication({"identifier": f"id_{i}"}) for i in range(10)]
        atl = Atlas(pubs)
        atl.save(path)

    def test_save_load_atlas_10(self, tmp_path):

        path = tmp_path / atlas_10_dir
        path.mkdir()

        pubs = [Publication({"identifier": f"id_{i}"}) for i in range(10)]
        atl = Atlas(pubs)
        atl.save(path)

        atl_loaded = Atlas.load(path)
        assert atl.publications == atl_loaded.publications


single_pub_bibtex_fp = "src/tests/data/single_publication.bib"
ten_pub_bibtex_fp = "src/tests/data/ten_publications.bib"
class TestAtlasBibtex:

    """Test loading atlases from bibtex files."""

    def test_from_bibtex_single(self):

        # Load expected values
        bibtex_fp = single_pub_bibtex_fp
        with open(bibtex_fp, "r") as f:
            bib_database = bibtexparser.load(f)
        entry: dict = bib_database.entries[0]        

        # Construct Atlas
        atl = Atlas.from_bibtex(single_pub_bibtex_fp)

        # Repeat checks from TestPublication.test_from_bibtex
        pub, = atl.publications

        # main attributes
        assert pub.identifier in entry.values()
        assert pub.abstract == entry["abstract"]
        assert pub.publication_date.year == int(entry["year"])

        # other attributes
        assert pub.url == entry["url"]
        assert pub.issn == entry["issn"]
        assert pub.doi == entry["doi"]
        assert pub.title == entry["title"]        

    def test_from_bibtex_10(self):

        # Load expected values
        bibtex_fp = ten_pub_bibtex_fp
        with open(bibtex_fp, "r") as f:
            bib_database = bibtexparser.load(f)

        # Construct Atlas
        atl = Atlas.from_bibtex(ten_pub_bibtex_fp)

        # Test length
        assert len(bib_database.entries) == len(atl)

        for entry in bib_database.entries:
            
            identifier = entry["doi"] # this is a test too
            pub = atl[identifier]

            # main attributes
            assert pub.abstract == entry["abstract"]
            assert pub.publication_date.year == int(entry["year"])

            # other attributes
            if "url" in entry:
                assert pub.url == entry["url"]
            assert pub.issn == entry["issn"]
            assert pub.doi == entry["doi"]
            assert pub.title == entry["title"]
