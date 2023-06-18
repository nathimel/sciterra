"""Test basic pipeline functionality with semantic scholar."""

class TestSemanticScholarAtlas:


    def test_semanticscholar_atlas_single_query(self):
        # construct an atlas w a single identifier, 
        pass

    def test_semanticscholar_atlas_100_query(self):
        # construct an atlas w a 100 identifiers
        # although in practice this will be done from expansions, 
        # so need to decide whether to just test expand
        pass

class TestSemanticScholarCartography:

    def test_semanticscholar_single_expand(self):
        # given a singleton atlas, expand by getting the most similar publication from the list of possible expansion ids
        pass

    def test_semanticscholar_100_expand(self):
        # given a singleton atlas, expand by getting up to 100 
        # of the _nearest_ publications.
        pass
