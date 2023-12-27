import numpy as np
import pandas as pd
import plotnine as pn

from datetime import datetime, date

from sciterra.mapping.atlas import Atlas
from sciterra.mapping.cartography import Cartographer
from sciterra.vectorization.scibert import SciBERTVectorizer


def main():
    # atlas_dir = "outputs/atlas_from_cc_region_8/"
    # atlas_dir = "outputs/atlas_s2-7-29-23_centered_imeletal"
    atlas_dir = "outputs/atlas_s2-11-11-23_bow-centered_hafenetal"

    atl = Atlas.load(atlas_dir)

    print(len(atl))

    vectorizer = SciBERTVectorizer(device="mps")
    crt = Cartographer(vectorizer=vectorizer)

    measurements = crt.measure_topography(atl, metrics=["density", "edginess"])

    citations_per_year = [
        atl[id].citation_count / (2023 - atl[id].publication_date.year)
        if (atl[id].publication_date.year < 2023 and atl[id].citation_count is not None)
        else 0.0
        for id in atl.projection.index_to_identifier
    ]
    # what if we just drop all those with 0 citations (per year)?
    # and those > 100 anyway
    citations_per_year = [
        item if (item > 0.0 and item < 100.0) else None for item in citations_per_year
    ]

    df = pd.DataFrame(
        measurements,
        columns=["density", "edginess"],
    )
    df["citations_per_year"] = citations_per_year
    df.dropna(inplace=True)  # not sure why this didn't take care of later NaNs
    df.to_csv("sciterra_data_from_cc_region_8.csv", index=False)


if __name__ == "__main__":
    main()
