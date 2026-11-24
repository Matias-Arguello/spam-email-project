import email
import re
from html.parser import HTMLParser
from string import punctuation

from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class HTMLStripper(HTMLParser):
    """Extrae únicamente el texto plano de un fragmento HTML."""

    def __init__(self):
        super().__init__()
        self.data = []

    def handle_data(self, data):
        self.data.append(data)


class EmailParser:
    """Normaliza el cuerpo de un correo electrónico para su vectorización.

    El pipeline aplicado es: extraer el cuerpo del mensaje, quitar tags HTML
    y URLs, tokenizar, y eliminar stopwords/puntuación aplicando stemming.
    """

    _PUNCTUATION = set(punctuation) | {"\n", "\t"}

    def __init__(self, language="english"):
        self._stemmer = PorterStemmer()
        self._stopwords = set(stopwords.words(language))

    def parse(self, raw_email):
        body = " ".join(self.get_body(raw_email))
        body = self.strip_tags(body)
        body = self.remove_urls(body)
        tokens = word_tokenize(body)
        tokens = self.clean_tokens(tokens)
        return " ".join(tokens)

    def get_body(self, raw_email):
        message = email.message_from_string(raw_email)
        return self._parse_payload(message.get_payload())

    def _parse_payload(self, payload):
        if isinstance(payload, str):
            return [payload]
        if isinstance(payload, list):
            body = []
            for part in payload:
                body += self._parse_payload(part.get_payload())
            return body
        return []

    def strip_tags(self, text):
        stripper = HTMLStripper()
        stripper.feed(text)
        return "".join(stripper.data)

    def remove_urls(self, text):
        return re.sub(r"http\S+", "", text)

    def clean_tokens(self, tokens):
        return [
            self._stemmer.stem(token)
            for token in tokens
            if token not in self._stopwords and token not in self._PUNCTUATION
        ]
