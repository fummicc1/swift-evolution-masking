from collections import Counter
import os
from spacy.language import Language
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_histogram_of_words(nlp: Language, markdown_text: str) -> Counter:
    doc = nlp(markdown_text)
    return Counter(word.text.lower() for word in doc if word.is_alpha)


def visualize_histogram(histogram: Counter, write_to_file: bool = False):
    common_words = [
        "the",
        "is",
        "in",
        "to",
        "of",
        "and",
        "a",
        "that",
        "it",
        "with",
        "as",
        "but",
        "if",
        "or",
        "because",
        "until",
        "while",
    ]
    df = pd.DataFrame(histogram.items(), columns=["word", "frequency"])
    # Remove common words to focus on the terms.
    df = df[~df["word"].isin(common_words)]
    df = df.sort_values(by="frequency", ascending=False)
    df.plot(kind="bar", x="word", y="frequency")
    plt.show()
    if write_to_file:
        os.makedirs("artifacts", exist_ok=True)
        plt.savefig("artifacts/word_freq_hist.png")
