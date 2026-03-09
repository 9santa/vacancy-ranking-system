import re

# Basic text cleaning:
# - lowercase
# - delete extra symbols
# - collapse spaces
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text) # leave eng, numbers, spaces, +, #, .
    text = re.sub(r"\s+", " ", text).strip()

    return text
