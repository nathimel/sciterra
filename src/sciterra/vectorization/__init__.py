from .gpt2 import GPT2Vectorizer
from .scibert import SciBERTVectorizer
from .sbert import SBERTVectorizer
from .word2vec import Word2VecVectorizer
from .bow import BOWVectorizer

vectorizers = {
    "GPT2": GPT2Vectorizer,
    "SciBERT": SciBERTVectorizer,
    "SBERT": SBERTVectorizer,
    "Word2Vec": Word2VecVectorizer,
    "BOW": BOWVectorizer,
}
