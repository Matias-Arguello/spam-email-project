# 📧 Spam Email Classifier

Clasificador de correos electrónicos spam/ham (no spam) basado en Machine Learning, entrenado sobre el [2007 TREC Public Spam Corpus](https://plg.uwaterloo.ca/~gvcormac/treccorpus07/about.html), un conjunto real de 75.419 correos etiquetados manualmente.

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikit-learn&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## Resultado

Sobre una muestra de 6.000 correos (80% train / 20% test, split estratificado), el modelo alcanza:

| Métrica | Valor |
|---|---|
| Accuracy | 0.991 |
| F1-score (spam) | 0.99 |
| F1-score (ham) | 0.98 |

```
              precision    recall  f1-score   support

         ham       0.99      0.98      0.98       302
        spam       0.99      1.00      0.99       898
```

## Enfoque

El pipeline de procesamiento de cada correo es:

1. **Parseo del mensaje** (`email.message_from_string`) para extraer el cuerpo, soportando correos `multipart/MIME`.
2. **Limpieza de texto**: eliminación de tags HTML y URLs.
3. **Tokenización y normalización** (NLTK): eliminación de stopwords y puntuación, aplicando *stemming* (algoritmo de Porter).
4. **Vectorización** del texto resultante con `CountVectorizer` (bag-of-words).
5. **Clasificación** con `LogisticRegression` de scikit-learn.

## Estructura del proyecto

```
spam-email-project/
├── pyproject.toml
├── requirements.txt
├── src/
│   └── spam_detector/
│       ├── email_parser.py     # Parseo y limpieza de correos (clase EmailParser)
│       ├── dataset.py          # Carga del corpus TREC07p
│       └── classifier.py       # SpamClassifier: vectorizador + modelo
├── scripts/
│   ├── train.py                 # Entrena el modelo y lo guarda en models/
│   └── predict.py                # Clasifica un correo usando el modelo entrenado
├── models/
│   └── spam_classifier.joblib    # Modelo ya entrenado (listo para usar)
└── data/                          # Dataset (no versionado, ver más abajo)
```

## Dataset

Por su tamaño (~1 GB), **el dataset no está incluido en este repositorio** (solo se necesita para reentrenar el modelo). Para reproducirlo:

1. Descargalo desde Kaggle: [Emails for spam or ham classification (Trec 2007)](https://www.kaggle.com/datasets/bayes2003/emails-for-spam-or-ham-classification-trec-2007).
2. Descomprimilo dentro de `data/`, de forma que quede la carpeta `data/trec07p/` con `full/index` y `data/` (los correos) en su interior.

## Instalación

```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

pip install -e .
python -m nltk.downloader stopwords punkt punkt_tab
```

## Uso

El repositorio incluye un modelo ya entrenado (`models/spam_classifier.joblib`), así que podés clasificar un correo sin necesidad de descargar el dataset:

```bash
python scripts/predict.py ruta/a/un/correo.txt
# -> spam  o  ham
```

El archivo debe contener un correo crudo en formato RFC822 (encabezados + cuerpo), igual a como se guardan los mensajes del corpus TREC07p.

### Reentrenar el modelo

Si descargaste el dataset (ver sección anterior):

```bash
python scripts/train.py --limit 6000
```

Opciones disponibles: `--index` (ruta al índice del dataset), `--limit` (cantidad de correos a usar) y `--model-out` (dónde guardar el modelo resultante).

## Posibles mejoras

- Probar `TfidfVectorizer` en lugar de `CountVectorizer`.
- Comparar con otros algoritmos habituales en clasificación de texto (`MultinomialNB`, `SVM`).
- Entrenar sobre el corpus completo (75.419 correos) en lugar de una muestra.
- Agregar tests automatizados para `EmailParser` y `SpamClassifier`.

## Licencia

Este proyecto está bajo la licencia MIT. Ver [LICENSE](LICENSE) para más detalles.
