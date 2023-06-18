"""Main container object for a large library of publications. Can be thought of as a vocabulary used in NLP, in that it stores a bidirectional mapping of strings to integers for indexing embeddings.
"""

from publication import Publication

class Atlas:

    def __init__(self, publications: list[Publication]) -> None:

        self.publications = publications
        self.embeddings = None
        
        # lookups for embeddings
        self.index_to_identifier = [str(pub) for pub in publications]
        self.identifier_to_index = {pub: i for i, pub in self.index_to_identifier}

    ######################################################################
    # Lookups for embeddings
    ######################################################################
    
    def __getitem__(self, identifier: str) -> int:
        """Get the index of a publication given its identifier.
        
        Raises:
            ValueError: the identifier is not in the Atlas.
        """
        if identifier in self.index_to_publications:
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

    def save_to_file(self, filepath: str) -> None:
        """Write the Atlas to a file.

        Args:
            filepath: path of file to save to.
        """
        pass
    
    @classmethod
    def load_from_file(cls, filepath: str, **kwargs):
        """Load an Atlas object from a saved file.

        Args:
            filepath: file with vocab, assumed output from `save_to_file`
        """
        pass

    ######################################################################
    # Other
    ######################################################################    

    def __len__(self) -> int:
        """Get length of the Atlas. """
        return len(self.index_to_publications)
