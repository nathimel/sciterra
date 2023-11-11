from .scibert import SciBERTVectorizer
from .sbert import SBERTVectorizer

vectorizers = {
    "SciBERT": SciBERTVectorizer,
    "SBERT": SBERTVectorizer,
}
