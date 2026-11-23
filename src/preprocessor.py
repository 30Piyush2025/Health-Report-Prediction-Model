import re

def clean_clinical_note(text: str) -> str:
    """
    Cleans medical note text by normalizing whitespace, stripping special tokens,
    and removing administrative artifacts.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\[\*\*.*?\*\*\]', ' ', text)  # Remove MIMIC de-identification brackets
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.,;:/\-%]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_age(text: str) -> int:
    """
    Extracts patient age using clinical regex patterns (e.g., '72-year-old', 'age 64').
    """
    patterns = [
        r'(\d{1,3})\s*(?:-|–|\s)?(?:year|yr|yo|y/o)',
        r'(?:age|aged)\s*(?:is|:)?\s*(\d{1,3})',
        r'(\d{1,3})\s*years\s*old'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            age = int(match.group(1))
            if 0 <= age <= 120:
                return age
    return 60  # Clinical median fallback

def extract_gender(text: str) -> str:
    """
    Extracts patient biological sex from clinical narrative.
    """
    lower = text.lower()
    if re.search(r'\b(female|woman|lady|girl|she|her)\b', lower):
        return 'Female'
    elif re.search(r'\b(male|man|gentleman|boy|he|his)\b', lower):
        return 'Male'
    return 'Unknown'
