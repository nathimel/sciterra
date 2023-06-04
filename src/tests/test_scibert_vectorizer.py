import numpy as np
from scipy.spatial.distance import cosine

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
    v1 = TestSciBERTVectorizer.vectorizer.embed_documents([abstract_str])
    v2 = TestSciBERTVectorizer.vectorizer.embed_documents([abstract_str])

    # flatten each (1, 768) into (768)
    v1 = v1.flatten()
    v2 = v2.flatten()

    # check identity
    assert np.all(v1 == v2)


def test_single_cosine_pair():
    v1 = TestSciBERTVectorizer.vectorizer.embed_documents([abstract_str])
    v2 = TestSciBERTVectorizer.vectorizer.embed_documents([abstract_str])

    # flatten each (1, 768) into (768)
    v1 = v1.flatten()
    v2 = v2.flatten()
    
    # Check that the cosine sim of doc w/ itself is 1
    sim = float(1 - cosine(v1, v2))
    assert sim == 1.0
