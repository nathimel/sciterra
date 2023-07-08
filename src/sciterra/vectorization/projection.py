import numpy as np

class Projection:
    """Basic wrapper for document embeddings and helper methods."""

    def __init__(
        self, 
        identifier_to_index: dict[str, int],
        index_to_identifier: tuple[str],
        embeddings: np.ndarray,
        ) -> None:
        """Construct a Projection object, a bidirectional mapping from identifiers to document embeddings.

        Args:
            identifiers_to_indices: a map from Publication identifiers to indices in the embedding matrix.

            indices_to_identifiers: a map from embedding indices to Publication identifiers.

            embeddings: ndarray of document embeddings of shape `(num_pubs, embedding_dim)`
        """
        self.identifier_to_index = identifier_to_index
        self.index_to_identifier = index_to_identifier
        self.embeddings = embeddings

    def indices_to_identifiers(self, indices) -> list[str]:
        """Retrieve the identifiers for a list of embedding matrix indices."""
        return [self.index_to_identifier[index] for index in indices]

    def identifiers_to_embeddings(self, identifiers: list[str]) -> np.ndarray:
        """Retrieve the document embeddings for a list of identifiers."""
        return [self.identifier_to_embedding(identifier) for identifier in identifiers]
    
    def identifier_to_embedding(self, identifier: str) -> np.ndarray:
        """Retrieve the document embedding of a Publication."""
        try: 
            embedding = self.embeddings[self.identifier_to_index[identifier]]
        except (IndexError, KeyError) as e:
            breakpoint()
        return embedding
        # return self.embeddings[self.identifier_to_index[identifier]]
    
    def __len__(self) -> int:
        return len(self.identifier_to_index)
