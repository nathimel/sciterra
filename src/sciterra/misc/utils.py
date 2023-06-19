"""Miscellaneous helper functions."""

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
