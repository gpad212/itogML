# app/preprocess.py
import re
import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from urllib.parse import urlparse

# Убедимся, что NLTK данные ищутся в известной директории
NLTK_DATA_DIR = os.environ.get("NLTK_DATA", "/usr/local/share/nltk_data")
nltk.data.path.append(NLTK_DATA_DIR)

# Попытаемся загрузить необходимые ресурсы (silently)
for resource in ('punkt', 'wordnet', 'stopwords', 'omw-1.4'):
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        try:
            nltk.download(resource, quiet=True, download_dir=NLTK_DATA_DIR)
        except Exception:
            # если нет интернета — продолжим, но lemmatizer может падать позже
            pass

tokenizer_pattern = re.compile(r'[A-Za-z]+')
wnl = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = set()

def clean_and_lemmatize(text: str) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ''
    # Оставляем только буквы (для URL полезно далее добавить доменные токены)
    tokens = tokenizer_pattern.findall(text)
    lemmas = []
    for t in tokens:
        low = t.lower()
        try:
            lemma = wnl.lemmatize(low)
        except Exception:
            lemma = low
        if lemma and lemma not in stop_words:
            lemmas.append(lemma)
    return ' '.join(lemmas)

def extract_domain_features(url: str) -> str:
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc or parsed.path  # иногда URL без схемы
        parts = [p for p in netloc.split('.') if p]
        subdomain = parts[0] if len(parts) > 2 else ''
        main = parts[-2] if len(parts) >= 2 else (parts[0] if parts else '')
        tld = parts[-1] if parts else ''
        path = parsed.path.replace('/', ' ')
        return ' '.join([subdomain, main, tld, path])
    except Exception:
        return ''

def preprocess_url(url: str) -> str:
    cleaned = clean_and_lemmatize(url)
    domain_feats = extract_domain_features(url)
    return (cleaned + ' ' + domain_feats).strip()
