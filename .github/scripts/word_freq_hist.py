from collections import Counter
import os
from spacy.language import Language
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_histogram_of_words(nlp: Language, markdown_text: str) -> Counter:
    doc = nlp(markdown_text)
    return Counter(word.text.lower() for word in doc if word.is_alpha and word.pos_ == "NOUN")


def visualize_histogram(histogram: Counter, write_to_file: bool = False):
    df = pd.DataFrame(histogram.items(), columns=["word", "frequency"])
    df = df.sort_values(by="frequency", ascending=False)
    # Only show the top 100 words.
    df = df.head(100)
    df.plot(kind="bar", x="word", y="frequency", figsize=(20, 15))
    plt.show()
    if write_to_file:
        os.makedirs("artifacts", exist_ok=True)
        plt.savefig("artifacts/word_freq_hist.png")
