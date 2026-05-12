"""
Data Cleaning Module
====================
Handles all text preprocessing steps before building the Bigram model.
Steps: lowercase → remove punctuation → tokenize → remove stopwords (optional)
"""

import re
import string


# Optional: basic English stopwords to filter out common words
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "he", "him", "his", "she", "her", "hers", "it", "its",
    "they", "them", "their", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "a", "an", "the",
    "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "into", "through", "to", "from",
    "in", "out", "on", "off", "over", "under", "then", "once", "so", "no",
}


class DataCleaning:
    """
    Cleans and preprocesses raw text for NLP tasks.

    Example usage:
        cleaner = DataCleaning()
        tokens = cleaner.clean("Hello, World! This is NLP.")
        # → ['hello', 'world', 'this', 'is', 'nlp']
    """

    def __init__(self, remove_stopwords: bool = False):
        """
        Parameters
        ----------
        remove_stopwords : bool
            If True, removes common English stopwords after tokenization.
            Default is False to preserve sentence structure for Bigram.
        """
        self.remove_stopwords = remove_stopwords

    def to_lowercase(self, text: str) -> str:
        """Step 1: Convert all characters to lowercase."""
        return text.lower()

    def remove_punctuation(self, text: str) -> str:
        """Step 2: Remove punctuation and special characters, keep only letters and spaces."""
        text = re.sub(r"[^\w\s]", " ", text)   # replace punctuation with space
        text = re.sub(r"\d+", "", text)          # remove digits
        text = re.sub(r"\s+", " ", text)         # collapse multiple spaces
        return text.strip()

    def tokenize(self, text: str) -> list[str]:
        """Step 3: Split cleaned text into individual word tokens."""
        return text.split()

    def filter_stopwords(self, tokens: list[str]) -> list[str]:
        """Step 4 (optional): Remove stopwords from token list."""
        return [t for t in tokens if t not in STOPWORDS]

    def clean(self, text: str) -> list[str]:
        """
        Full pipeline: runs all cleaning steps in order.

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        list[str]
            List of cleaned tokens.
        """
        text = self.to_lowercase(text)
        text = self.remove_punctuation(text)
        tokens = self.tokenize(text)

        if self.remove_stopwords:
            tokens = self.filter_stopwords(tokens)

        return tokens

    def clean_corpus(self, sentences: list[str]) -> list[str]:
        """
        Clean an entire corpus (list of sentences) and return all tokens.

        Parameters
        ----------
        sentences : list[str]
            List of raw sentences.

        Returns
        -------
        list[str]
            Combined list of all cleaned tokens from all sentences.
        """
        all_tokens = []
        for sentence in sentences:
            all_tokens.extend(self.clean(sentence))
        return all_tokens

