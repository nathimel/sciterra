import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_distances

from tqdm import tqdm

from sciterra.vectorization.scibert import SciBERTVectorizer

abstract_str = "We use cosmological hydrodynamic simulations with stellar feedback from the FIRE (Feedback In Realistic Environments) project to study the physical nature of Lyman limit systems (LLSs) at z ≤ 1. At these low redshifts, LLSs are closely associated with dense gas structures surrounding galaxies, such as galactic winds, dwarf satellites and cool inflows from the intergalactic medium. Our analysis is based on 14 zoom-in simulations covering the halo mass range M<SUB>h</SUB> ≈ 10<SUP>9</SUP>-10<SUP>13</SUP> M<SUB>⊙</SUB> at z = 0, which we convolve with the dark matter halo mass function to produce cosmological statistics. We find that the majority of cosmologically selected LLSs are associated with haloes in the mass range 10<SUP>10</SUP> ≲ M<SUB>h</SUB> ≲ 10<SUP>12</SUP> M<SUB>⊙</SUB>. The incidence and H I column density distribution of simulated absorbers with columns in the range 10^{16.2} ≤ N_{H I} ≤ 2× 10^{20} cm<SUP>-2</SUP> are consistent with observations. High-velocity outflows (with radial velocity exceeding the halo circular velocity by a factor of ≳ 2) tend to have higher metallicities ([X/H] ∼ -0.5) while very low metallicity ([X/H] &lt; -2) LLSs are typically associated with gas infalling from the intergalactic medium. However, most LLSs occupy an intermediate region in metallicity-radial velocity space, for which there is no clear trend between metallicity and radial kinematics. The overall simulated LLS metallicity distribution has a mean (standard deviation) [X/H] = -0.9 (0.4) and does not show significant evidence for bimodality, in contrast to recent observational studies, but consistent with LLSs arising from haloes with a broad range of masses and metallicities."


class TestSciBERTVectorizer:

    vectorizer = SciBERTVectorizer()

def test_single_vector():
    embedding = TestSciBERTVectorizer.vectorizer.embed_documents([abstract_str])

    # Check embedding is of correct type, shape, and has no nans
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1, 768)
    assert not np.isnan(embedding).any()


def test_identity_of_embeddings():
    embeddings = TestSciBERTVectorizer.vectorizer.embed_documents([abstract_str, abstract_str])
    # check identity
    assert np.all( embeddings[0] == embeddings[1] )


def test_single_cosine_pair():
    embeddings = TestSciBERTVectorizer.vectorizer.embed_documents([abstract_str, abstract_str])
    
    # Check that the cosine sim of doc w/ itself is 1
    sim = float(1 - cosine(embeddings[0], embeddings[1]))
    assert sim == 1.0

def test_basic_cosine_matrix():
    # TODO: this takes way too long. 
    # like pair above, but pretending that we have more than 2 publications.
    num_pubs = 10 
    embeddings = np.array([
        TestSciBERTVectorizer.vectorizer.embed_documents([abstract_str] * num_pubs).flatten()
    ])
    cosine_matrix = cosine_distances(embeddings, embeddings)
    assert np.all( cosine_matrix == 0 )
