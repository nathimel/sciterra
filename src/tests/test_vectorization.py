import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_distances

from tqdm import tqdm

from sciterra.vectorization.vectorizer import Vectorizer
from sciterra.vectorization import scibert, sbert, word2vec

astro_corpus_1 = "src/tests/data/corpora/example.txt"
model_path_1 = "src/tests/data/models/word2vec_model_example.model"

abstract_str = "We use cosmological hydrodynamic simulations with stellar feedback from the FIRE (Feedback In Realistic Environments) project to study the physical nature of Lyman limit systems (LLSs) at z ≤ 1. At these low redshifts, LLSs are closely associated with dense gas structures surrounding galaxies, such as galactic winds, dwarf satellites and cool inflows from the intergalactic medium. Our analysis is based on 14 zoom-in simulations covering the halo mass range M<SUB>h</SUB> ≈ 10<SUP>9</SUP>-10<SUP>13</SUP> M<SUB>⊙</SUB> at z = 0, which we convolve with the dark matter halo mass function to produce cosmological statistics. We find that the majority of cosmologically selected LLSs are associated with haloes in the mass range 10<SUP>10</SUP> ≲ M<SUB>h</SUB> ≲ 10<SUP>12</SUP> M<SUB>⊙</SUB>. The incidence and H I column density distribution of simulated absorbers with columns in the range 10^{16.2} ≤ N_{H I} ≤ 2× 10^{20} cm<SUP>-2</SUP> are consistent with observations. High-velocity outflows (with radial velocity exceeding the halo circular velocity by a factor of ≳ 2) tend to have higher metallicities ([X/H] ∼ -0.5) while very low metallicity ([X/H] &lt; -2) LLSs are typically associated with gas infalling from the intergalactic medium. However, most LLSs occupy an intermediate region in metallicity-radial velocity space, for which there is no clear trend between metallicity and radial kinematics. The overall simulated LLS metallicity distribution has a mean (standard deviation) [X/H] = -0.9 (0.4) and does not show significant evidence for bimodality, in contrast to recent observational studies, but consistent with LLSs arising from haloes with a broad range of masses and metallicities."  # 252 tokens

##############################################################################
# SciBERT
##############################################################################


class TestSciBERTVectorizer:
    vectorizer = scibert.SciBERTVectorizer()  # don't pass 'mps' for CI tests
    embedding_dim = scibert.EMBEDDING_DIM

    def test_single_vector(self):
        embedding = TestSciBERTVectorizer.vectorizer.embed_documents([abstract_str])[
            "embeddings"
        ]

        # Check embedding is of correct type, shape, and has no nans
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, TestSciBERTVectorizer.embedding_dim)
        assert not np.isnan(embedding).any()

    def test_identity_of_embeddings(self):
        embeddings = TestSciBERTVectorizer.vectorizer.embed_documents(
            [abstract_str, abstract_str]
        )["embeddings"]
        # check identity
        assert np.all(embeddings[0] == embeddings[1])

    def test_single_cosine_pair(self):
        embeddings = TestSciBERTVectorizer.vectorizer.embed_documents(
            [abstract_str, abstract_str]
        )["embeddings"]

        # Check that the cosine sim of doc w/ itself is 1
        # n.b., see sklearn.metrics.pairwise.cosine_similarity
        sim = float(1 - cosine(embeddings[0], embeddings[1]))
        assert sim == 1.0

    def test_basic_cosine_matrix(self):
        # like pair above, but pretending that we have more than 2 publications.

        num_pubs = 300
        # n.b., 1000 typically takes 83.75s with mps; Colab cuda takes just 29s
        # github actions just uses cpu, so maybe don't waste time stress testing

        embeddings = np.array(
            [
                TestSciBERTVectorizer.vectorizer.embed_documents(
                    [abstract_str] * num_pubs
                )["embeddings"].flatten()
            ]
        )
        cosine_matrix = cosine_distances(embeddings, embeddings)
        assert np.all(cosine_matrix == 0)


##############################################################################
# SBERT
##############################################################################


class TestSBERTVectorizer:
    vectorizer = sbert.SBERTVectorizer()  # don't pass 'mps' for CI tests
    embedding_dim = sbert.EMBEDDING_DIM

    def test_single_vector(self):
        embedding = TestSBERTVectorizer.vectorizer.embed_documents([abstract_str])[
            "embeddings"
        ]

        # Check embedding is of correct type, shape, and has no nans
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, TestSBERTVectorizer.embedding_dim)
        assert not np.isnan(embedding).any()

    def test_identity_of_embeddings(self):
        embeddings = TestSBERTVectorizer.vectorizer.embed_documents(
            [abstract_str, abstract_str]
        )["embeddings"]
        # check identity
        assert np.all(embeddings[0] == embeddings[1])

    def test_single_cosine_pair(self):
        embeddings = TestSBERTVectorizer.vectorizer.embed_documents(
            [abstract_str, abstract_str]
        )["embeddings"]

        # Check that the cosine sim of doc w/ itself is 1
        # n.b., see sklearn.metrics.pairwise.cosine_similarity
        sim = float(1 - cosine(embeddings[0], embeddings[1]))
        assert sim == 1.0

    def test_basic_cosine_matrix(self):
        # like pair above, but pretending that we have more than 2 publications.

        num_pubs = 300
        # n.b., 1000 typically takes 83.75s with mps; Colab cuda takes just 29s
        # github actions just uses cpu, so maybe don't waste time stress testing

        embeddings = np.array(
            [
                TestSBERTVectorizer.vectorizer.embed_documents(
                    [abstract_str] * num_pubs
                )["embeddings"].flatten()
            ]
        )
        cosine_matrix = cosine_distances(embeddings, embeddings)
        assert np.all(cosine_matrix == 0)


##############################################################################
# Word2Vec
##############################################################################


class TestWord2VecVectorizer:
    corpus_path = astro_corpus_1
    model_path = model_path_1
    vectorizer = word2vec.Word2VecVectorizer(
        corpus_path=corpus_path,
        model_path=model_path,
    )
    embedding_dim = word2vec.EMBEDDING_DIM

    def test_single_vector(self):
        embedding = TestWord2VecVectorizer.vectorizer.embed_documents([abstract_str])[
            "embeddings"
        ]

        # Check embedding is of correct type, shape, and has no nans
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, TestWord2VecVectorizer.embedding_dim)
        assert not np.isnan(embedding).any()

    def test_failed_embedding(self):
        result = TestWord2VecVectorizer.vectorizer.embed_documents(
            ["outofvocabularyitem"]
        )

        # Check embedding is of correct type, shape, and has no nans
        assert np.array_equal(result["fail_indices"], np.array([0]))
        assert np.array_equal(result["success_indices"], np.array([]))
        assert np.array_equal(result["embeddings"], np.array([]))

    def test_identity_of_embeddings(self):
        embeddings = TestWord2VecVectorizer.vectorizer.embed_documents(
            [abstract_str, abstract_str]
        )["embeddings"]
        # check identity
        assert np.all(embeddings[0] == embeddings[1])

    def test_single_cosine_pair(self):
        embeddings = TestWord2VecVectorizer.vectorizer.embed_documents(
            [abstract_str, abstract_str]
        )["embeddings"]

        # Check that the cosine sim of doc w/ itself is 1
        # n.b., see sklearn.metrics.pairwise.cosine_similarity
        sim = float(1 - cosine(embeddings[0], embeddings[1]))
        assert sim == 1.0

    def test_basic_cosine_matrix(self):
        # like pair above, but pretending that we have more than 2 publications.

        num_pubs = 300

        embeddings = np.array(
            [
                TestWord2VecVectorizer.vectorizer.embed_documents(
                    [abstract_str] * num_pubs
                )["embeddings"].flatten()
            ]
        )
        cosine_matrix = cosine_distances(embeddings, embeddings)
        assert np.all(cosine_matrix == 0)
