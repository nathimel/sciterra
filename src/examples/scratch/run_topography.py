"""Notebook not working so we'll try and compute and save the measurements to a npy file."""

import numpy as np
import pandas as pd
import plotnine as pn

from sciterra.mapping.atlas import Atlas
from sciterra.mapping.cartography import Cartographer
from sciterra.vectorization.scibert import SciBERTVectorizer

atlas_dir = "outputs/atlas_from_cc_region_8/"

atl = Atlas.load(atlas_dir)

vectorizer = SciBERTVectorizer(device="mps")
crt = Cartographer(vectorizer=vectorizer)

# results in bus error
measurements = crt.measure_topography(atl, metrics=["density", "edginess"])

np.save(measurements, "measurements.npy")
