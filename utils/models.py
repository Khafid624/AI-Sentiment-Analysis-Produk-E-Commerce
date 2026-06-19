import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def train_and_compare(df, text_col, label_col="sentiment"):

    # =====================
    # DATA
    # =====================
    X = df[text_col].astype(str)
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    # =====================
    # MODELS
    # =====================
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC()
    }

    results = []

    for name, model in models.items():

        model.fit(X_train_vec, y_train)
        pred = model.predict(X_test_vec)

        acc = accuracy_score(y_test, pred)

        results.append({
            "model": name,
            "accuracy": acc
        })

    return pd.DataFrame(results)