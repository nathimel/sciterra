"""Simple preprocessing of scientific abstracts prior to vectorization."""

import spacy

nlp = spacy.load("en_core_web_sm")

# Another off the shelf simple tokenizer
from gensim.utils import simple_preprocess


def custom_preprocess(
    document: str,
    allowed_pos_tags: set = {"NOUN", "VERB", "ADJ"},
) -> list[str]:
    """Get all of the lemmas of the words in a document, filtering by POS.

    Args:
        document: a multi-sentence string

        allowed_pos_tags: keep and lemmatize words that are tagged as one of these POS categories.

    Returns:
        a list of the lemmatized, filtered words in the document

    Given the domain-specificity, we choose to heuristically stem instead of performing full, linguistically precise lemmatization that would require detailed vocabulary rules. That said, the nltk WordNet lemmatizer doesn't immediately seem to do better than basic stemming

    See https://github.com/zhafen/cc/blob/master/cc/utils.py#L173.
    """
    return [
        token.lemma_
        for sent in nlp(document).sents
        for token in sent
        if token.pos_ in allowed_pos_tags
    ]
