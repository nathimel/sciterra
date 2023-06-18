"""Test basic Atlas functionality, independently of API. To obtain realistic publication data, should probably read in a .bib file."""


from sciterra.atlas import Atlas

class TestAtlasEmpty:

    def test_create_empty_atlas(self):

        atl = Atlas()

    def test_load_empty_atlas(self):

        # create an empty file

        # load an atlas from it

        # delete the empty file
        pass

    def test_save_empty_atlas(self):

        # cerate empty atlas

        # save it
        pass

class TestAtlasSingle:
    """Test an Atlas with a single publication."""

    def test_atlas_single(self):
        pass

    def test_load_atlas_single(self):
        pass

    def test_save_atlas_single(self):
        pass


class TestAtlas100:

    def test_atlas_100(self):
        pass

    def test_load_atlas_100(self):
        pass

    def test_save_atlas_100(self):
        pass

