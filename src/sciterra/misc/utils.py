"""Miscellaneous helper functions."""

import pickle
import time
from functools import wraps
from requests.exceptions import ReadTimeout, ConnectionError

# For Bibtex parsing


def standardize_month(month: str) -> str:
    month = month.lower()

    if len(month) == 3:
        standardized = {
            "jan": "January",
            "feb": "February",
            "mar": "March",
            "apr": "April",
            "may": "May",
            "jun": "June",
            "jul": "July",
            "aug": "August",
            "sep": "September",
            "oct": "October",
            "nov": "November",
            "dec": "December",
        }[month]
    else:
        standardized = {
            "january": "January",
            "february": "February",
            "march": "March",
            "april": "April",
            "may": "May",
            "june": "June",
            "july": "July",
            "august": "August",
            "september": "September",
            "october": "October",
            "november": "November",
            "december": "December",
        }[month]

    return standardized


# For querying apis


def keep_trying(
    n_attempts=5,
    allowed_exceptions=(ReadTimeout, ConnectionError),
    verbose=True,
    sleep_after_attempt=1,
):
    """Sometimes we receive server errors. We don't want that to disrupt the entire process, so this decorator allow trying n_attempts times.

    ## API_extension::get_data_via_api
    ## This decorator is general, except for the default allowed exception.

    Args:
        n_attempts (int):
            Number of attempts before letting the exception happen.

        allowed_exceptions (tuple of class):
            Allowed exception class. Set to BaseException to keep trying regardless of exception.

        sleep_after_attempt (int):
            Number of seconds to wait before trying each additional attempt.

        verbose (bool):
            If True, be talkative.

    Example Usage:
        > @keep_trying( n_attempts=4 )
        > def try_to_call_web_api():
        >     " do stuff "
    """

    def _keep_trying(f):
        @wraps(f)
        def wrapped_fn(*args, **kwargs):
            # Loop over for n-1 attempts, trying to return
            for i in range(n_attempts - 1):
                # waiting may help with connection errors?
                time.sleep(sleep_after_attempt)

                try:
                    result = f(*args, **kwargs)
                    if i > 0 and verbose:
                        print(f"Had to call {f} {i+1} times to get a response.")
                    return result
                # TODO: need to branch for every case.
                except allowed_exceptions as _:
                    continue

            # On last attempt just let it be
            if verbose:
                print(
                    f"Had to call {f} {n_attempts} times to get a response. Trying once more."
                )
            return f(*args, **kwargs)

        return wrapped_fn

    return _keep_trying


def chunk_ids(ids: list[str], call_size=2000):
    """Helper function to chunk bibcodes or paperIds into smaller sublists if appropriate."""
    # Break into chunks
    assert (
        call_size <= 2000
    ), "Max number of calls ExportQuery can handle at a time is 2000."
    if len(ids) > call_size:
        chunked_ids = [ids[i : i + call_size] for i in range(0, len(ids), call_size)]
    else:
        chunked_ids = [
            ids,
        ]

    return chunked_ids


# File IO
def write_pickle(fn: str, data):
    with open(fn, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(fn: str):
    with open(fn, "rb") as f:
        data = pickle.load(f)
    return data
