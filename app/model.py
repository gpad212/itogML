# app/model.py
import os
import joblib
import pandas as pd
from app.preprocess import preprocess_url

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECT_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
os.makedirs("/app/output", exist_ok=True)

_model = None
_vectorizer = None

def load_model():
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
            raise FileNotFoundError(f"Model files not found in {MODEL_DIR}. Please run training and place model.pkl and vectorizer.pkl there.")
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECT_PATH)
    return _model, _vectorizer

def predict_from_csv(csv_path: str) -> str:
    df = pd.read_csv(csv_path)
    if 'url' not in df.columns:
        raise ValueError("CSV должен содержать столбец 'url'")
    df['processed_url'] = df['url'].apply(preprocess_url)
    model, vectorizer = load_model()
    X = vectorizer.transform(df['processed_url'].fillna(''))
    preds = model.predict(X)
    df['prediction'] = preds
    output_path = os.path.join("/app/output", f"predictions_{os.path.basename(csv_path)}")
    df.to_csv(output_path, index=False)
    return output_path
