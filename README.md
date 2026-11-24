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

Ver el detalle completo (incluyendo la matriz de confusión) en [`notebooks/spam_detection.ipynb`](notebooks/spam_detection.ipynb).

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
├── notebooks/
│   └── spam_detection.ipynb   # Análisis, entrenamiento y evaluación del modelo
├── src/
│   └── spam_detector/
│       ├── email_parser.py     # Parseo y limpieza de correos (clase EmailParser)
│       └── dataset.py          # Carga del corpus TREC07p
├── data/                        # Dataset (no versionado, ver más abajo)
├── requirements.txt
└── LICENSE
```

## Dataset

Por su tamaño (~1 GB), **el dataset no está incluido en este repositorio**. Para reproducir el proyecto:

1. Descargalo desde Kaggle: [Emails for spam or ham classification (Trec 2007)](https://www.kaggle.com/datasets/bayes2003/emails-for-spam-or-ham-classification-trec-2007).
2. Descomprimilo dentro de `data/`, de forma que quede la carpeta `data/trec07p/` con `full/index` y `data/` (los correos) en su interior.

## Instalación

```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

pip install -r requirements.txt

python -m nltk.downloader stopwords punkt punkt_tab
```

## Uso

```bash
jupyter notebook notebooks/spam_detection.ipynb
```

O usando los módulos directamente:

```python
import sys
sys.path.insert(0, "src")

from spam_detector import EmailParser, read_raw_emails
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

emails, labels = read_raw_emails("data/trec07p/full/index", limit=6000)
parser = EmailParser()
processed = [parser.parse(e) for e in emails]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(processed)

clf = LogisticRegression(max_iter=1000)
clf.fit(X, labels)
```

## Posibles mejoras

- Probar `TfidfVectorizer` en lugar de `CountVectorizer`.
- Comparar con otros algoritmos habituales en clasificación de texto (`MultinomialNB`, `SVM`).
- Entrenar sobre el corpus completo (75.419 correos) en lugar de una muestra.
- Serializar el modelo entrenado (`joblib`) para reutilizarlo sin reentrenar.

## Licencia

Este proyecto está bajo la licencia MIT. Ver [LICENSE](LICENSE) para más detalles.
