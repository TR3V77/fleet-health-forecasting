"""
Auto-tags support tickets by category from their free-text description alone,
instead of relying on a human to pick a dropdown value.

Trains a TF-IDF + logistic regression (SGDClassifier, log loss) text classifier on the
historical tickets (description -> category), reports held-out accuracy plus a per-category
breakdown to outputs/classifier_report.txt, and saves the trained pipeline to data/.

To swap in an LLM instead of this classical model, replace the body of `predict()` with an
API call that prompts the model with the category list and the ticket description and asks
for one label back. `main()` evaluates through `predict()`, so the evaluation stays the same.
"""
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_PATH = DATA_DIR / "ticket_categorizer.pkl"


def predict(pipeline: Pipeline, texts: list[str]) -> list[str]:
    return list(pipeline.predict(texts))


def main():
    tickets = pd.read_csv(DATA_DIR / "tickets.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        tickets["description"], tickets["category"], test_size=0.25, random_state=42, stratify=tickets["category"]
    )

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("clf", SGDClassifier(loss="log_loss", random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = predict(pipeline, X_test)
    report = classification_report(y_test, y_pred)
    print("Held-out classification report (predicting category from free-text description):\n")
    print(report)
    (OUTPUT_DIR / "classifier_report.txt").write_text(
        "Held-out classification report (25% test split, category predicted from description text)\n\n" + report
    )

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Saved trained classifier -> {MODEL_PATH}")

    print("\nExample predictions on new, unseen-style descriptions:")
    samples = [
        "Battery drains to zero within an hour of unplugging the charger.",
        "Printer keeps jamming on every duplex job this week.",
        "Laptop won't connect to the office VPN since this morning.",
    ]
    for text, pred in zip(samples, predict(pipeline, samples)):
        print(f"  '{text}' -> {pred}")


if __name__ == "__main__":
    main()
