import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from app.preprocess import preprocess_url

# Скачиваем нужные ресурсы NLTK (если не скачаны)
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("stopwords")
nltk.download("punkt")

# === 1. Загружаем данные ===
df = pd.read_csv("train.csv")

# Проверим, что нужные колонки есть
required_columns = {"Id", "url", "Predicted"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Файл должен содержать колонки: {required_columns}")

# === 2. Предобработка ===
df["processed_url"] = df["url"].astype(str).apply(preprocess_url)

# === 3. Разделение на обучающую и тестовую выборку ===
X_train, X_test, y_train, y_test = train_test_split(
    df["processed_url"], df["Predicted"], test_size=0.2, random_state=42
)

# === 4. Векторизация ===
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === 5. Обучение модели ===
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# === 6. Оценка точности ===
accuracy = model.score(X_test_vec, y_test)
print(f"✅ Точность модели: {accuracy:.4f}")

# === 7. Сохраняем модель и векторизатор ===
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Модель и векторизатор успешно сохранены!")
