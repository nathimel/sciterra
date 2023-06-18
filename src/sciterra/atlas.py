"""Main container object for a large library of publications. Can be thought of as a vocabulary used in NLP, in that it stores a bidirectional mapping of strings to integers for indexing embeddings.
"""

import os
import warnings
import numpy as np
import pandas as pd

from .publication import FIELDS, Publication

class Atlas:

    def __init__(
        self, 
        publications: list[Publication], 
        ) -> None:

        self.publications = sorted(list(set(publications)))
        if self.publications:
            self.id_to_pub = {str(pub): pub for pub in self.publications}

    ######################################################################
    # Lookup    ######################################################################
    
    def __getitem__(self, identifier: str) -> Publication:
        """Get a publication given its identifier.
        
        Raises:
            ValueError: the identifier is not in the Atlas.
        """
        if identifier in self.id_to_pub:
            return self.id_to_pub[identifier]
        raise ValueError(f"Identifier {identifier} not in Atlas.")
    
    ######################################################################
    # File I/O
    ######################################################################

    def save(
        self, 
        atlas_dirpath: str, 
        publications_fn: str = "publications.csv", 
        overwrite_publications: bool = True,
        ) -> None:
        """Write the Atlas to a directory containing a CSV file of publications and a .npy file of embeddings.

        Args:
            atlas_dirpath: path of directory to save files to.

            publications_fn: name of file to save publications to.

            overwrite_publications: whether to overwrite an existing publications file.

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
    
    @classmethod
    def load(
        cls, 
        atlas_dirpath: str, 
        publications_fn: str = "publications.csv", 
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
        
        return cls(publications)

    ######################################################################
    # Other
    ######################################################################    

    def __len__(self) -> int:
        """Get length of the Atlas."""
        return len(self.publications)
