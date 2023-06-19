"""Miscellaneous helper functions."""

# For Bibtex parsing

def standardize_month(month: str) -> str:
    month = month.lower()
    return {
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
