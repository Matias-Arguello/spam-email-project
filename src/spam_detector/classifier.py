import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .email_parser import EmailParser


class SpamClassifier:
    """Clasificador spam/ham: vectorización bag-of-words + regresión logística."""

    def __init__(self):
        self._parser = EmailParser()
        self.vectorizer = CountVectorizer()
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, processed_texts, labels):
        X = self.vectorizer.fit_transform(processed_texts)
        self.model.fit(X, labels)

    def evaluate(self, processed_texts, labels):
        X = self.vectorizer.transform(processed_texts)
        predictions = self.model.predict(X)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "report": classification_report(labels, predictions),
            "confusion_matrix": confusion_matrix(labels, predictions),
        }

    def predict_raw(self, raw_email):
        """Clasifica un correo sin procesar (texto crudo en formato RFC822)."""
        processed = self._parser.parse(raw_email)
        X = self.vectorizer.transform([processed])
        return self.model.predict(X)[0]

    def save(self, path):
        joblib.dump({"vectorizer": self.vectorizer, "model": self.model}, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        instance = cls()
        instance.vectorizer = data["vectorizer"]
        instance.model = data["model"]
        return instance
