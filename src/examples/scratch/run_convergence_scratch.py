import numpy as np
import pandas as pd
import plotnine as pn
import matplotlib.pyplot as plt

from sciterra.mapping.atlas import Atlas
from sciterra.mapping.cartography import Cartographer
from sciterra.vectorization.scibert import SciBERTVectorizer

atlas_dir = "/Users/nathanielimel/uci/projects/sciterra/src/examples/scratch/outputs/atlas_s2-11-10-23_centered_hafenetal"

def main():

    atl = Atlas.load(atlas_dir)    

    kernels = atl.history['kernel_size']

    con_d = 3
    kernel_size = 10
    converged_filter = kernels[:, -con_d] >= kernel_size
    ids = np.array(atl.projection.index_to_identifier)
    converged_pub_ids = ids[converged_filter]

    crt = Cartographer(vectorizer=SciBERTVectorizer()) 

    measurements = crt.measure_topography(
        atl, 
        ids=converged_pub_ids,
        metrics=["density", "edginess"], 
        kernel_size=10,
    )


if __name__ == "__main__":
    main()