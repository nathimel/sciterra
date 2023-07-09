"""Test basic Atlas functionality, independently of API. To obtain realistic publication data, should probably read in a .bib file."""

import bibtexparser

import pandas as pd

from sciterra.atlas import Atlas
from sciterra.publication import Publication

from sciterra.misc.utils import write_pickle, read_pickle


atlas_empty_dir = "atlas_empty"
atlas_single_dir = "atlas_single"
atlas_10_dir = "atlas_10"


class TestAtlasDummy:
    def test_create_empty_atlas(self):
        atl = Atlas([])
        assert atl.publications == dict()

    def test_save_empty_atlas(self, tmp_path):
        path = tmp_path / atlas_empty_dir
        path.mkdir()
        atl = Atlas([])
        atl.save(path)

    def test_save_load_empty_atlas(self, tmp_path):
        path = tmp_path / atlas_empty_dir
        path.mkdir()
        embedding_path = path / "publications.pkl"
        write_pickle(embedding_path, [])

        atl = Atlas.load(path)
        assert atl.publications == Atlas([]).publications

    """Test an Atlas with a single publication."""

    def test_atlas_single(self):
        pub = Publication({"identifier": "id"})
        atl = Atlas([pub])
        assert atl[str(pub)] == pub

    def test_atlas_single_duplicate(self):
        pub = Publication({"identifier": "id"})
        atl = Atlas([pub, pub])
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


class TestAtlasCitationNetwork:

    """Check the basic graph structure encoded into an Atlas by each publication's citations and references."""

    def test_singleton_network(self):
        pass

    def test_three_node_network(self):
        pass
