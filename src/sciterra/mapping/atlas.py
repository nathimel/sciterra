"""Main container object for a large library of publications.
"""

import os
import warnings
import numpy as np
import pandas as pd

from typing import Any

from .publication import Publication
from ..vectorization.projection import Projection
from ..misc.utils import read_pickle, write_pickle, get_verbose, custom_formatwarning

warnings.formatwarning = custom_formatwarning


class Atlas:

    """Data structure for storing publications.

    `self.projection`: the Projection object containing the embeddings of all publications and their mapping to str identifiers.

    `self.bad_ids`: a list of identifiers that have failed for some reason or other during an expansion, and will be excluded from subsequent expansions.

    `self.history`: dict of the form {'pubs_per_update': list[list[str]], 'kernel_size': np.ndarray of ints of shape `(num_pubs, last_update)` where last_update <= the total number of expansions performed.}

    `self.center`: the core, central Publication identifier repeatedly passed to `cartography.Cartographer.expand`. Default is None, which means the Atlas has no internal record of the central publication.
    """

    def __init__(
        self,
        publications: list[Publication],
        projection: Projection = None,
        bad_ids: set[str] = set(),
        history: dict[str, Any] = None,
        center: str = None,
    ) -> None:
        if not isinstance(publications, list):
            raise ValueError

        self.publications: dict[str, Publication] = {
            str(pub): pub for pub in publications
        }
        self.projection = projection
        self.bad_ids = bad_ids
        self.history = history
        self.center = center

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

    @property
    def ids(self) -> list[str]:
        """Get a list of all the publication identifiers in the Atlas."""
        return list(self.publications.keys())

    ######################################################################
    # File I/O
    ######################################################################

    def save(
        self,
        atlas_dirpath: str,
        overwrite: bool = True,
    ) -> None:
        """Write the Atlas to a directory containing a .pkl binary for each attribute.

        Warnings cannot be silenced.

        Args:
            atlas_dirpath: path of directory to save files to.
        """

        # Create directory as needed, or overwrite existing files
        if os.path.isdir(atlas_dirpath):
            if not overwrite:
                warnings.warn(f"Skipping overwrite of atlas data found at {atlas_dirpath}.")
                return
        else:
            warnings.warn(f"Recursively creating atlas data directory at {atlas_dirpath}.")
            os.makedirs(atlas_dirpath)

        attributes = {
            k: getattr(self, k)
            for k in [
                "publications",
                "projection",
                "bad_ids",
                "history",
                "center",
            ]
        }

        for attribute in attributes:
            if getattr(self, attribute) is not None:
                # write the list version to be consistent with load and constructor
                if attribute == "publications":
                    attributes[attribute] = list(self.publications.values())

                fn = f"{attribute}.pkl"
                fp = os.path.join(atlas_dirpath, fn)
                if os.path.isfile(fp):
                    warnings.warn(f"Overwriting existing file at {fp}.")
                else:
                    warnings.warn(f"Writing to {fp}.")
                write_pickle(fp, attributes[attribute])
            else:
                warnings.warn(f"No {attribute} to save, skipping.")

    @classmethod
    def load(
        cls,
        atlas_dirpath: str,
    ):
        """Load an Atlas object from a directory containing the .pkl binary for each attribute.

        Warnings cannot be silenced.

        Args:
            atlas_dirpath: directory where .pkl binaries will be read from

        """
        attributes = {
            k: None
            for k in [
                "publications",
                "projection",
                "bad_ids",
                "history",
                "center",
            ]
        }
        for attribute in attributes:
            fn = f"{attribute}.pkl"
            fp = os.path.join(atlas_dirpath, fn)
            if os.path.isfile(fp):
                attributes[attribute] = read_pickle(fp)
            else:
                warnings.warn(f"No {attribute} to read, skipping.")

        if attributes["publications"] is None:
            warnings.warn("Loading empty atlas.")
            attributes["publications"] = list()

        return cls(**{k: v for k, v in attributes.items() if v is not None})

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
