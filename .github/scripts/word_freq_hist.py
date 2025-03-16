from collections import Counter
import os
from spacy.language import Language
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import boto3
from typing import Dict, List, Set, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import tqdm


def get_histogram_of_words(nlp: Language, markdown_text: str) -> Counter:
    """
    Count frequency of nouns in the given markdown text.
    
    Args:
        nlp: spaCy language model
        markdown_text: Text to analyze
        
    Returns:
        Counter object with words and their frequencies
    """
    doc = nlp(markdown_text)
    return Counter(
        word.text.lower() for word in doc if word.is_alpha and word.pos_ == "NOUN"
    )


def visualize_histogram_and_return_df(
    histogram: Counter, write_to_file: bool = False
) -> pd.DataFrame:
    """
    Create a DataFrame and visualization from word histogram.
    
    Args:
        histogram: Counter with word frequencies
        write_to_file: Whether to save outputs to file
        
    Returns:
        DataFrame with words and frequencies
    """
    df = pd.DataFrame(histogram.items(), columns=["word", "frequency"])
    df = df.sort_values(by="frequency", ascending=False)
    # Only show the top 100 words on the plot.
    top_df = df.head(100)
    top_df.plot(kind="bar", x="word", y="frequency", figsize=(20, 15))
    plt.show(block=False)
    if write_to_file:
        os.makedirs("artifacts", exist_ok=True)
        plt.savefig("artifacts/word_freq_hist_top_100.png")
        # Save DataFrame as JSON
        df.to_json("artifacts/word_freq_hist.json", orient="records")
    return df


class WordSimilarityCalculator:
    """Calculates and manages word similarities using word embeddings."""
    
    def __init__(self, nlp: Language):
        """
        Initialize with a spaCy language model.
        
        Args:
            nlp: spaCy language model
        """
        self.nlp = nlp
        self.embeddings = {}
        self.similarity_map = {}
        
    def build_embeddings(self, words: Set[str]) -> None:
        """
        Build embeddings for a set of words.
        
        Args:
            words: Set of words to generate embeddings for
        """
        print(f"Building embeddings for {len(words)} words...")
        self.embeddings = {}
        
        for i, word in enumerate(words):
            if i % 100 == 0:
                print(f"Processing word {i}/{len(words)}...")
                
            doc = self.nlp(word)
            if doc.has_vector and doc.vector_norm > 0:
                self.embeddings[word] = doc.vector
                
        print(f"Generated embeddings for {len(self.embeddings)} out of {len(words)} words")
    
    def calculate_similarities(self, batch_size: int = 100) -> Dict[str, List[str]]:
        """
        Calculate similarity between words using batch processing for better performance.
        
        Args:
            batch_size: Size of batches for processing
            
        Returns:
            Dictionary mapping words to their most similar words
        """
        print("Calculating word similarities...")
        words = list(self.embeddings.keys())
        total_words = len(words)
        self.similarity_map = {}
        
        # Process in batches for better performance
        for i in range(0, total_words, batch_size):
            batch_words = words[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(total_words + batch_size - 1)//batch_size}...")
            
            # Create batch of embeddings
            batch_embeddings = np.array([self.embeddings[word] for word in batch_words])
            
            # Calculate similarities for all words in one go
            all_embeddings = np.array([self.embeddings[word] for word in words])
            
            # Calculate cosine similarity matrix for the batch
            # Each row is a word in the batch, each column is a word in the vocabulary
            similarities = cosine_similarity(batch_embeddings, all_embeddings)
            
            # Process results for each word in batch
            for idx, word in enumerate(batch_words):
                # Get similarity scores for this word
                sim_scores = similarities[idx]
                
                # Create list of (word, score) pairs, excluding the word itself
                word_scores = [(w, sim_scores[j]) 
                               for j, w in enumerate(words) 
                               if w != word]
                
                # Sort by similarity (highest first)
                word_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Store top 50 most similar words
                self.similarity_map[word] = [w for w, _ in word_scores[:50]]
        
        return self.similarity_map
    
    def get_similar_words(self, word: str, n: int = 50) -> List[str]:
        """
        Get n most similar words for a given word.
        
        Args:
            word: Word to find similar words for
            n: Number of similar words to return
            
        Returns:
            List of similar words
        """
        if word not in self.similarity_map:
            return []
        return self.similarity_map[word][:n]


def build_word_similarity_map(nlp: Language, nouns: Set[str]) -> Dict[str, List[str]]:
    """
    Build a map of each word to its 50 most similar words.
    
    Args:
        nlp: spaCy language model
        nouns: Set of all nouns found in the corpus
        
    Returns:
        Dictionary mapping each word to a list of similar words
    """
    calculator = WordSimilarityCalculator(nlp)
    calculator.build_embeddings(nouns)
    similarity_map = calculator.calculate_similarities()
    return similarity_map
