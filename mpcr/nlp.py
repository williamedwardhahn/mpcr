"""Bag-of-words helpers for the chatbot notebooks. NLTK and its tokenizer data load on first use."""
import numpy as np

from ._lazy import LazyAttr

__all__ = ['tokenize', 'stem', 'bag_of_words', 'stemmer', 'PorterStemmer']

PorterStemmer = LazyAttr('nltk.stem.porter', 'PorterStemmer')
stemmer = LazyAttr('nltk.stem.porter', 'PorterStemmer', call=True)   # a PorterStemmer(), made on first use
_ready = False


def _punkt():
    """Download NLTK's sentence tokenizer once (newer NLTK calls it punkt_tab)."""
    global _ready
    if _ready:
        return
    import nltk
    for name in ('punkt', 'punkt_tab'):
        try:
            nltk.data.find(f'tokenizers/{name}')
        except LookupError:
            nltk.download(name, quiet=True)
    _ready = True


def tokenize(sentence):
    """Split a sentence into words, punctuation and numbers."""
    _punkt()
    import nltk
    return nltk.word_tokenize(sentence)


def stem(word):
    """The root form of a word: organize, organizes, organizing -> organ."""
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """1.0 for each vocabulary word that appears (after stemming) in the sentence, else 0.0.

    bag_of_words(['hello', 'how', 'are', 'you'], ['hi', 'hello', 'I', 'you', 'bye'])
    -> [0, 1, 0, 1, 0]"""
    sentence_words = {stem(w) for w in tokenized_sentence}
    return np.array([1.0 if w in sentence_words else 0.0 for w in words], dtype=np.float32)
