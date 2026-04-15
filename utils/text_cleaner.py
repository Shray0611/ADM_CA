import re

def clean_text(text: str) -> str:
    """
    Cleans text by lowercasing, removing special characters, and removing extra spaces.
    NO stopwords removal, NO lemmatization.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove special characters (keep alphanumeric and space)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
