"""Helper functions for analyzing data yielded by an atlas."""

import pandas as pd
import numpy as np

from ..mapping.atlas import Atlas
from ..mapping.cartography import Cartographer
from ..vectorization.vectorizer import Vectorizer


def atlas_to_measurements(
    atl: Atlas,
    vectorizer: Vectorizer,
    con_d: float,
    kernel_size=16,  # TODO: find a principled way of selecting this value., i.e. Fig. 5 from Imel & Hafen (2023). https://openreview.net/pdf?id=mISayy7DPI.
    metrics: list[str] = ["density", "edginess"],
    fields_of_study=None,
    max_year: int = 2023,  # consider 2022
) -> pd.DataFrame:
    """Compute the density, edginess, and citations per year metrics for each publicaation in an atlas w.r.t. a vectorizer and convergence configurations, and return the results in a dataframe.

    Args:
        atl: the Atlas to measure

        vectorizer: the Vectorizer to use to compute density and edginess

        con_d: a reverse index. This represents a convergence degree, in the sense that it is the number of updates before the last udpate to require that a publication's neighborhood has not changed identity of composition.

        This will be used to compute the inverse index for the second axis of the array `atl.history['kernel_size']`, representing the degree of convergence. For details about this array see `sciterra.mapping.cartography.Cartographer.converged_kernel_size`. Default is 1, which means we will filter to all publications that have not changed neighborhoods up to `kernel_size` up until the very last update. If 2, then up to the second to last update, etc.

        kernel_size: the minimum required size of the neighborhood that we will require to not have changed, w.r.t. `cond_d`. Default is 16.

        metrics: the list of str names corresponding to metrics to compute for the atlas. See `sciterra.mapping.topography` for possible metrics.

    """
    kernels = atl.history[
        "kernel_size"
    ]  # shape `(num_pubs, max_update)`, where `max_update` is typically the total number of updates if this function is called after the atlas has been sufficiently built out.
    max_update = kernels.shape[1]

    # Get all publications that have not changed neighborhoods up to kernel_size for the last con_d updates
    converged_filter = kernels[:, -con_d] >= kernel_size
    ids = np.array(atl.projection.index_to_identifier)
    converged_pub_ids = ids[converged_filter]

    print(
        f"Convergence degree {con_d} (out of {max_update} total updates) yields {len(converged_pub_ids)} ids out of {len(atl)} total ids."
    )

    # Optionally filter only to `field_of_study` publications
    if fields_of_study is not None:
        converged_pub_ids = [
            id
            for id in converged_pub_ids
            if any(fld in atl[id].fields_of_study for fld in atl[id].fields_of_study)
        ]

    # Compute density, edginess
    crt = Cartographer(
        vectorizer=vectorizer,
    )
    measurements = crt.measure_topography(
        atl,
        ids=converged_pub_ids,
        metrics=metrics,
        kernel_size=kernel_size,
    )

    # Get citations
    citations_per_year = [
        atl[id].citation_count / (max_year - atl[id].publication_date.year)
        if (
            atl[id].publication_date.year < max_year
            and atl[id].citation_count is not None
        )
        else 0.0
        for id in converged_pub_ids
    ]

    # Annotate the center (this feels inefficient, but oh well)
    is_center = [identifier == atl.center for identifier in converged_pub_ids]

    if not any(is_center):
        import warnings

        warnings.warn(
            f"The center publication is not within the set of convered publications."
        )

    df = pd.DataFrame(
        measurements,
        columns=metrics,
    )
    df["citations_per_year"] = citations_per_year
    df["is_center"] = is_center

    df = df[~np.isinf(df["density"])]  # drop infs which occur for BOW vectorizer
    # TODO what about other very high densities that result from close to 0?

    df.dropna(
        inplace=True,
    )

    print(f"There are {len(df)} total observations after filtering.")

    return df
