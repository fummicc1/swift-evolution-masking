from collections import Counter
import os
from spacy.language import Language
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import boto3


def get_histogram_of_words(nlp: Language, markdown_text: str) -> Counter:
    doc = nlp(markdown_text)
    return Counter(
        word.text.lower() for word in doc if word.is_alpha and word.pos_ == "NOUN"
    )


def visualize_histogram_and_return_df(
    histogram: Counter, write_to_file: bool = False
) -> pd.DataFrame:
    df = pd.DataFrame(histogram.items(), columns=["word", "frequency"])
    df = df.sort_values(by="frequency", ascending=False)
    df = df.head(300)
    # Only show the top 100 words on the plot.
    top_df = df.head(100)
    top_df.plot(kind="bar", x="word", y="frequency", figsize=(20, 15))
    plt.show()
    if write_to_file:
        os.makedirs("artifacts", exist_ok=True)
        plt.savefig("artifacts/word_freq_hist_top_100.png")
        # Save DataFrame as JSON
        df.to_json("artifacts/word_freq_hist.json", orient="records")
    return df
