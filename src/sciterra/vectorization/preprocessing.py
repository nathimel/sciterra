"""Simple preprocessing of scientific abstracts prior to vectorization."""

import spacy

# Another off the shelf simple tokenizer
from gensim.utils import simple_preprocess


class CustomPreprocessor:
    def __init__(
        self,
        allowed_pos_tags: set = {"NOUN", "VERB", "ADJ"},
        model="en_core_web_sm",
    ) -> None:
        """Initialize a custom tokenizer.

        Args:

            allowed_pos_tags: keep and lemmatize words that are tagged as one of these POS categories.

            model: the name of the spacy language model to load, assuming it is already downloaded.
        """
        try:
            nlp = spacy.load(model)
        except OSError:
            raise OSError(
                f"Can't find model '{model}'; make sure you have run 'python3 -m spacy download {model}'!"
            )

        self.nlp = nlp
        self.allowed_pos_tags = allowed_pos_tags

    def custom_preprocess(
        self,
        document: str,
    ) -> list[str]:
        """Get all of the lemmas of the words in a document, filtering by POS.

        Args:
            document: a multi-sentence string

        Returns:
            a list of the lemmatized, filtered words in the document

        Given the domain-specificity, we choose to heuristically stem instead of performing full, linguistically precise lemmatization that would require detailed vocabulary rules. That said, the nltk WordNet lemmatizer doesn't immediately seem to do better than basic stemming

        See https://github.com/zhafen/cc/blob/master/cc/utils.py#L173.
        """
        return [
            token.lemma_
            for sent in self.nlp(document).sents
            for token in sent
            if token.pos_ in self.allowed_pos_tags
        ]
