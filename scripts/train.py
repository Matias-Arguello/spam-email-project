import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from spam_detector import SpamClassifier, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena el clasificador de spam sobre el corpus TREC07p."
    )
    parser.add_argument(
        "--index",
        default="data/trec07p/full/index",
        help="Ruta al archivo de índice del dataset",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=6000,
        help="Número de correos a usar para entrenar y evaluar",
    )
    parser.add_argument(
        "--model-out",
        default="models/spam_classifier.joblib",
        help="Ruta donde guardar el modelo entrenado",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    processed_emails, labels = build_dataset(args.index, limit=args.limit)

    X_train, X_test, y_train, y_test = train_test_split(
        processed_emails, labels, test_size=0.2, stratify=labels, random_state=42
    )

    classifier = SpamClassifier()
    classifier.fit(X_train, y_train)

    metrics = classifier.evaluate(X_test, y_test)
    print(f"\nAccuracy: {metrics['accuracy']:.3f}\n")
    print(metrics["report"])
    print("Matriz de confusión:")
    print(metrics["confusion_matrix"])

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(model_out)
    print(f"\nModelo guardado en {model_out}")


if __name__ == "__main__":
    main()
