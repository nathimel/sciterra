"""Main container object for a large library of publications.
"""

import os
import warnings
import numpy as np
import pandas as pd

from typing import Any

from .publication import Publication
from .vectorization.projection import Projection
from .misc.utils import read_pickle, write_pickle


class Atlas:
    def __init__(
        self,
        publications: list[Publication],
        projection: Projection = None,
    ) -> None:
        if not isinstance(publications, list):
            raise ValueError
        self.publications: dict[str, Publication] = {
            str(pub): pub for pub in publications
        }
        self.projection: Projection = projection

    ######################################################################
    # Lookup    ######################################################################

    def __getitem__(self, identifier: str) -> Publication:
        """Get a publication given its identifier.

        Raises:
            ValueError: the identifier is not in the Atlas.
        """
        if identifier in self.publications:
            return self.publications[identifier]
        raise ValueError(f"Identifier {identifier} not in Atlas.")

    ######################################################################
    # File I/O
    ######################################################################

    def save(
        self,
        atlas_dirpath: str,
        publications_fn: str = "publications.pkl",
        projection_fn: str = "projection.pkl",
        overwrite_publications: bool = True,
        overwrite_projection: bool = True,
    ) -> None:
        """Write the Atlas to a directory containing a CSV file of publications and a .npy file of embeddings.

        Write the Atlas to a directory containing a .pkl file of publications and a .pkl file of the projection.

        Args:
            atlas_dirpath: path of directory to save files to.

            publications_fn: name of file to save publications to.

            projection_fn: name of file to save projection to.

            overwrite_publications: whether to overwrite an existing publications file.

            overwrite_projection: whether to overwrite an existing projection file.

        """
        # save publications
        if self.publications and overwrite_publications:
            fp = os.path.join(atlas_dirpath, publications_fn)
            if os.path.isfile(fp):
                warnings.warn(f"Overwriting existing file at {fp}.")
            write_pickle(
                fp, list(self.publications.values())
            )  # write the list version to be consistent with load and constructor
        else:
            warnings.warn("No publications to save, skipping.")

        # save projection
        if self.projection is not None and overwrite_projection:
            fp = os.path.join(atlas_dirpath, projection_fn)
            if os.path.isfile(fp):
                warnings.warn(f"Overwriting existing file at {fp}.")
            write_pickle(fp, self.projection)
        else:
            warnings.warn("No projection to save, skipping.")

    @classmethod
    def load(
        cls,
        atlas_dirpath: str,
        publications_fn: str = "publications.pkl",
        projection_fn: str = "projection.pkl",
        **kwargs,
    ):
        """Load an Atlas object from a directory containing publications and/or their projection.

        Args:
            atlas_dirpath: file with vocab, assumed output from `save_to_file`

            publications_fn: name of file to load publications from.

            projection_fn: name of file to load projection from.
        """

        # load publications
        fp = os.path.join(atlas_dirpath, publications_fn)
        publications = None
        if os.path.isfile(fp):
            publications = read_pickle(fp)
        else:
            warnings.warn("No publications to read, skipping.")

        # load projection
        fp = os.path.join(atlas_dirpath, projection_fn)
        projection = None
        if os.path.isfile(fp):
            projection = read_pickle(fp)
        else:
            warnings.warn("No projection to read, skipping.")

        if publications is None:
            warnings.warn("Loading empty atlas.")
            publications = list()

        return cls(
            publications,
            projection,
        )

    ######################################################################
    # Other
    ######################################################################

    def __len__(self) -> int:
        """Get length of the Atlas."""
        return len(self.publications)

    def __eq__(self, __value: object) -> bool:
        return (
            self.publications == __value.publications
            and self.projection == __value.projection
        )
