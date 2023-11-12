from .scibert import SciBERTVectorizer
from .sbert import SBERTVectorizer
from .word2vec import Word2VecVectorizer

vectorizers = {
    "SciBERT": SciBERTVectorizer,
    "SBERT": SBERTVectorizer,
    "Word2Vec": Word2VecVectorizer,
}
