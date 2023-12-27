import argparse

import torch
import numpy as np


def set_seed(seed: int) -> None:
    """Sets various random seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Atlas expansion script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "--bibtex_fp",
        required=True,
        type=str,
        # default="data/hafenLowredshiftLymanLimit2017.bib",
        help="Bibtex file solely containing the initial (and center, if appropriate) publication.",
    )
    required_named.add_argument(
        "--atlas_dir",
        required=True,
        type=str,
        # default="outputs/default_atlas_dir",
        help="Atlas binary files will be saved/overwritten in this directory.",
    )

    parser.add_argument("--seed", type=int, default=42, help="Specify the random seed.")
    parser.add_argument(
        "--target_size",
        type=int,
        default=10000,
        help="The target size for an Atlas to attain via iterative expansion.",
    )
    parser.add_argument(
        "--max_pubs_per_expand",
        type=int,
        default=1000,
        help="The maximum number of publications to query an API for in one expand call. Increasing this number can help expansion not get stuck, but is also more likely to result in API connection failures and throttling.",
    )
    parser.add_argument(
        "--max_failed_expansions",
        type=int,
        default=2,
        help="The number of times to continue trying to iteratively expanding after Atlas successively gets no new publications.",
    )
    parser.add_argument(
        "--call_size",
        type=int,
        default=10,
        help="The number of papers to request from an API every call. Also, keep in mind rate limits. For example, Semantic Scholar allows up to 5,000 calls per 5 minutes; in practice, however, more than 10 calls per query typically results in connection errors and readtimeouts.",
    )
    parser.add_argument(
        "--centered",
        type=bool,
        default=True,
        help="Whether to retrieve publications in order of similarity to the center publication. If False, retrieves a random sample of citations and references in the Atlas accumulated from the previous iteration.",
    )
    parser.add_argument(
        "--api",
        type=str,
        choices=[
            "S2",
            "ADS",
        ],
        default="S2",
        help="The API, corresponding to a sciterra.librarian.Librarian, to use to retrieve publications.",
    )
    parser.add_argument(
        "--vectorizer",
        type=str,
        choices=[
            "SciBERT",
            "SBERT",
            "Word2Vec",
            "BOW",
        ],
        default="SciBERT",
        help="The vectorizer, corresponding to a sciterra.vectorization.Vectorizer, to use to get document embeddings for each publication abstract for retrieving (cosine) similar publications.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="The path to the file to save/load a vectorizer's trained model. Word2Vec by default trains on a large corpus of scientific text, so loading a pretrained model can save significant time.",
    )

    args = parser.parse_args()
    return args
