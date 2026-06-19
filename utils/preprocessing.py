import re
import string

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    return text