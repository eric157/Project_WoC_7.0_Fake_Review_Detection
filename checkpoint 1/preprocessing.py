import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import contractions

# Ensure NLTK dependencies are downloaded
nltk_dependencies = ['punkt', 'wordnet', 'stopwords', 'omw-1.4']
for dependency in nltk_dependencies:
    nltk.download(dependency, quiet=True)

def preprocess_text(text):
    """
    Preprocess text by applying the following steps:
    - Expand contractions
    - Convert to lowercase
    - Remove digits and punctuation
    - Tokenize and remove stopwords
    - Lemmatize tokens
    - Reconstruct the text
    """
    try:
        text = contractions.fix(text)
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    except Exception as e:
        return ""
