import argparse

from spam_detector import SpamClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clasifica un correo electrónico como spam o ham."
    )
    parser.add_argument(
        "email_file",
        help="Ruta a un archivo de texto con el correo crudo (formato RFC822)",
    )
    parser.add_argument(
        "--model",
        default="models/spam_classifier.joblib",
        help="Ruta al modelo entrenado (generado por scripts/train.py)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    classifier = SpamClassifier.load(args.model)

    with open(args.email_file, errors="ignore") as f:
        raw_email = f.read()

    prediction = classifier.predict_raw(raw_email)
    print(prediction)


if __name__ == "__main__":
    main()
