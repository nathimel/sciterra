"""Main container object for a large library of publications. Can be thought of as a vocabulary used in NLP, in that it stores a bidirectional mapping of strings to integers for indexing embeddings.
"""

import os
import warnings
import numpy as np
import pandas as pd

from publication import FIELDS
from publication import Publication

class Atlas:

    def __init__(
        self, 
        publications: list[Publication], 
        # embeddings: np.ndarray = None,
        ) -> None:

        self.publications = list(set(publications))
        # self.embeddings = embeddings
        
        # lookups for embeddings
        if self.publications:
            self.index_to_identifier = [str(pub) for pub in self.publications]
            self.identifier_to_index = {pub: i for i, pub in enumerate(self.index_to_identifier)}

            # if self.embeddings is not None:
            #     if len(self.embeddings) != len(self.publications):
            #         raise Exception(f"Number of embeddings ({len(self.embeddings)}) do not match number of publications ({self.publications}).")

    ######################################################################
    # Lookups for embeddings
    ######################################################################
    
    def __getitem__(self, identifier: str) -> Publication:
        """Get a publication given its identifier.
        
        Raises:
            ValueError: the identifier is not in the Atlas.
        """
        if identifier in self.identifier_to_index:
            return self.identifier_to_index[identifier]
        raise ValueError(f"Identifier {identifier} not in Atlas.")
    
    def identifiers_to_indices(self, identifiers: list[str]) -> list[int]:
        """Get all indices for a list of tokens."""
        return [self.identifier_to_index[identifer] for identifer in identifiers]

    def indices_to_identifiers(self, indices: list[int]) -> list[str]:
        """Get all identifiers for a list of integer indices."""
        return [self.index_to_identifier[idx] for idx in indices]
    
    ######################################################################
    # File I/O
    ######################################################################

    def save(
        self, 
        atlas_dirpath: str, 
        publications_fn: str = "publications.csv", 
        embeddings_fn: str = "embeddings.npy",
        overwrite_publications: bool = True,
        overwrite_embeddings: bool = True,
        ) -> None:
        """Write the Atlas to a directory containing a CSV file of publications and a .npy file of embeddings.

        Args:
            atlas_dirpath: path of directory to save files to.

            publications_fn: name of file to save publications to.

            embeddings_fn: name of file to save embeddings to.

            overwrite_publications: whether to overwrite an existing publications file.

            overwrite_embeddings: whether to overwrite an existing embeddings file.
        """
        # save publications
        if self.publications:
            if overwrite_publications:
                pub_data = pd.DataFrame(
                    data=[pub.to_csv_entry() for pub in self.publications],
                    columns=FIELDS,
                )
                fp = os.path.join(atlas_dirpath, publications_fn)
                if os.path.isfile(fp):
                    warnings.warn(f"Overwriting existing file at {fp}.")
                pub_data.to_csv(fp, index=False)
        else:
            warnings.warn("No publications to save, skipping.")

        # save embeddings
        if self.embeddings is not None:
            if overwrite_embeddings:
                fp = os.path.join(atlas_dirpath, embeddings_fn)
                if os.path.isfile(fp):
                    warnings.warn(f"Overwriting existing file at {fp}.")
                np.save(fp, self.embeddings)
        else:
            warnings.warn("No embeddings to save, skipping.")
    
    @classmethod
    def load(
        cls, 
        atlas_dirpath: str, 
        publications_fn: str = "publications.csv", 
        embeddings_fn: str = "embeddings.csv",
        **kwargs,
        ):
        """Load an Atlas object from a directory containing publications saved to a CSV file and possibly embeddings saved to a .npy file.

        Args:
            atlas_dirpath: file with vocab, assumed output from `save_to_file`
        """

        # load publications
        fp = os.path.join(atlas_dirpath, publications_fn)
        pub_data = pd.read_csv(fp)
        publications = [Publication.from_csv_entry(entry) for entry in pub_data.values.tolist()]

        # load embeddings
        fp = os.path.join(atlas_dirpath, embeddings_fn)
        embeddings = None
        if os.path.isfile(fp):
            embeddings = np.load(fp)
        
        return cls(publications, embeddings)

    ######################################################################
    # Other
    ######################################################################    

    def __len__(self) -> int:
        """Get length of the Atlas."""
        return len(self.index_to_publications)
