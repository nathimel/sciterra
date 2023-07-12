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
        return self.embeddings[self.identifier_to_index[identifier]]

    def __len__(self) -> int:
        return len(self.identifier_to_index)

    def __eq__(self, __value: object) -> bool:
        return (
            np.array_equal(self.embeddings, __value.embeddings)
            and self.identifier_to_index == __value.identifier_to_index
            and self.index_to_identifier == __value.index_to_identifier
        )


######################################################################
# Merge projections
######################################################################


def merge(proj_a: Projection, proj_b: Projection) -> Projection:
    """Return the result of merging projection `proj_a` with projection `proj_b`.

    This adds to proj_a all embedding data contained in proj_b that is missing from proj_a. This means that the resulting projection can only be greater or equal in size to proj_a.
    """
    if proj_b is None or not len(proj_b):
        return proj_a

    # Get the data in the new projection missing from the old
    indices_missing = []
    identifiers_missing = []
    for id, idx in proj_b.identifier_to_index.items():
        if proj_a is None or id not in proj_a.identifier_to_index:
            indices_missing.append(idx)
            identifiers_missing.append(id)

    # Get just the missing embeddings
    embeddings_missing = np.array(
        [
            embedding
            for idx, embedding in enumerate(proj_b.embeddings)
            if idx in set(indices_missing)
        ]
    )

    # Concatenate index mapping and embeddings
    idx_to_ids_new = identifiers_missing
    embeddings_new = embeddings_missing
    if proj_a is not None:
        idx_to_ids_new = list(proj_a.index_to_identifier) + idx_to_ids_new
        embeddings_new = np.concatenate([proj_a.embeddings, embeddings_new])

    # Return a new projection
    return Projection(
        identifier_to_index={id: idx for idx, id in enumerate(idx_to_ids_new)},
        index_to_identifier=tuple(idx_to_ids_new),
        embeddings=embeddings_new,
    )
