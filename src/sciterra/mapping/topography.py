"""Functions for measuring topographic properties of (the semantic feature space of publications inside) an Atlas."""

import numpy as np


########################################################################
# Density metrics
########################################################################


def smoothing_length_metric(
    idx: int,
    cospsi_matrix: np.ndarray,
    valid_indices: np.ndarray,
    kernel_size: int = 16,
):
    """Proxy for the density of a publication defined as the minimum
    arc length that encloses kernel_size other publications.

    Args:
        idx: the index of the vector to calculate the measurement for.

        cospsi_matrix: a 2D matrix of pairwise cosine similarity scores for publication embeddings.

        valid_indices: Indices of the other publication used when calculating the measurements.

        kernel_size: number of K nearest neighbors to calculate the measurement on.

    Returns:
        h: float representing arc length containing `kernel_size` other publications. (Assumes normalized to a radius of 1.)
    """

    # We can't have the kernel larger than the number of valid publications
    if kernel_size > len(valid_indices):
        return np.nan

    # Get 1D array of similarity scores to idx vector
    cospsi = cospsi_matrix[idx][valid_indices]

    # Get cosine distance to the least similar vector
    # np.sort orders from least to greatest similarity, so reverse after
    cospsi_max = np.sort(cospsi)[::-1][kernel_size - 1]

    # Compute arclength to furthest vector
    return np.arccos(cospsi_max)


def density_metric(
    idx: int,
    cospsi_matrix: np.ndarray,
    valid_indices: np.ndarray,
    kernel_size: int = 16,
):
    """Estimate the density of a publication by calculating the
    smoothing length that encloses kernel_size other publications.

    Args:
        idx: the index of the vector to calculate the measurement for.

        cospsi_matrix: a 2D matrix of pairwise cosine similarity scores for publication embeddings.

        valid_indices: Indices of the other publication used when calculating the measurements.

        kernel_size: number of K nearest neighbors to calculate the measurement on.

    Returns:
        density: a float representing `kernel_size` divided by arc length containing `kernel_size` other publications.
    """

    h = smoothing_length_metric(idx, cospsi_matrix, valid_indices, kernel_size)
    density = kernel_size / h

    # # TODO: there is serious numerical instability for BOW methods. Not sure what the most principled way to deal with them are.
    # if density > 1e7 and np.isfinite(density):
    #     breakpoint()

    return density


########################################################################
# Asymmetry metrics
########################################################################


def edginess_metric(
    idx: int,
    cospsi_matrix: np.ndarray,
    valid_indices: np.ndarray,
    publication_indices: np.ndarray,
    embeddings: np.ndarray,
    kernel_size: int = 16,
) -> float:
    """Estimate the asymmetry of a publication by calculating the difference between that publication's projection and the other publications within the kernel. Normalized to between 0 and 1.

    Args:
        idx: the index of the vector to calculate the measurement for.

        cospsi_matrix: an np.ndarray of shape `(num_pubs, num_pubs)` representing pairwise cosine similarity scores for publication embeddings.

        valid_indices: an np.ndarray of shape `(num_valid_pubs)` representing indices of the other publications used when calculating the measurements, i.e. num_valid_pubs <= num_pubs.

        publication_indices: an np.ndarray of shape `(num_pubs,)` representing indices of all publications in the atlas projection

        embeddings: an np.ndarray of shape `(num_pubs, embedding_dim)` vectors for all publications in the atlas projection

        kernel_size: number of K nearest neighbors to calculate the measurement on.

    Returns:
        a float representing the normalized magnitude of the asymmetry metric.

    """
    return (
        kernel_constant_asymmetry_metric(
            idx,
            cospsi_matrix,
            valid_indices,
            publication_indices,
            embeddings,
            kernel_size=kernel_size,
        )
        / kernel_size
    )


def kernel_constant_asymmetry_metric(
    idx: int,
    cospsi_matrix: np.ndarray,
    valid_indices: np.ndarray,
    publication_indices: np.ndarray,
    embeddings: np.ndarray,
    kernel_size: int = 16,
) -> float:
    """Estimate the asymmetry of a publication by calculating the difference
    between that publication's projection and the other publications within
    the kernel.

    Args:
        idx: an int representing the index of the vector to calculate the measurement for.

        cospsi_matrix: an np.ndarray of shape `(num_pubs, num_pubs)` representing pairwise cosine similarity scores for publication embeddings.

        valid_indices: an np.ndarray of shape `(num_valid_pubs)` representing indices of the other publications used when calculating the measurements, i.e. num_valid_pubs <= num_pubs.

        publication_indices: an np.ndarray of shape `(num_pubs,)` representing indices of all publications in the atlas projection

        embeddings: an np.ndarray of shape `(num_pubs, embedding_dim)` vectors for all publications in the atlas projection

        kernel_size: an int representing the number of K nearest neighbors to calculate the measurement on.

    Returns:
        mag: a float representing the magnitude of the asymmetry metric.
    """

    # We can't have the kernel larger than the number of valid publications
    if kernel_size > len(valid_indices):
        return np.nan

    # Input
    cospsi = cospsi_matrix[idx][valid_indices]
    sorted_inds = np.argsort(cospsi)[::-1][:kernel_size]
    other_inds = publication_indices[valid_indices][sorted_inds]
    embedding = embeddings[idx]
    other_embeddings = embeddings[other_inds]

    # Differences
    diff = embedding - other_embeddings
    diff_mag = np.linalg.norm(diff, axis=1)
    result = (diff / diff_mag[:, np.newaxis]).sum(axis=0)
    mag = np.linalg.norm(result)

    return mag
