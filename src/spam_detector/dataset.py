from pathlib import Path

from .email_parser import EmailParser


def read_raw_emails(index_path, limit=None):
    """Lee los correos crudos y sus etiquetas a partir del índice del TREC07p.

    El archivo de índice (``full/index``) contiene líneas con el formato
    ``<etiqueta> ../data/<archivo>``, con rutas relativas al directorio que
    contiene al propio índice (p. ej. ``trec07p/full/``).
    """
    index_path = Path(index_path)
    dataset_root = index_path.parent

    with open(index_path, "r") as f:
        lines = f.read().splitlines()

    if limit is not None:
        lines = lines[:limit]

    texts, labels = [], []
    for line in lines:
        label, relative_path = line.split(" ../")
        email_path = dataset_root / ".." / relative_path
        with open(email_path, errors="ignore") as f:
            texts.append(f.read())
        labels.append(label)

    return texts, labels


def build_dataset(index_path, limit=None, verbose=True):
    """Construye el dataset (texto procesado, etiqueta) listo para vectorizar."""
    parser = EmailParser()
    raw_texts, labels = read_raw_emails(index_path, limit)

    processed_texts = []
    for i, raw_text in enumerate(raw_texts):
        if verbose:
            print(f"\rProcesando correo {i + 1}/{len(raw_texts)}", end="")
        processed_texts.append(parser.parse(raw_text))

    if verbose:
        print()

    return processed_texts, labels
