"""
Bigram Model Module
===================
Builds a probability matrix from tokenized text and returns ranked
next-word suggestions given the last word typed by the user.

Architecture:
    Bigram(word1 → word2) probability =
        count(word1, word2) / count(word1)
"""

from collections import defaultdict
from data_cleaning import DataCleaning


class BigramModel:
    """
    Trains a Bigram language model on a corpus and provides
    next-word predictions with probability scores.

    Example usage:
        model = BigramModel()
        model.train(sentences)
        suggestions = model.suggest("the", n=5)
        # → [("quick", 0.33), ("lazy", 0.25), ...]
    """

    def __init__(self, remove_stopwords: bool = False):
        self.cleaner = DataCleaning(remove_stopwords=remove_stopwords)
        # bigram_counts[w1][w2] = count of (w1, w2) pairs seen
        self.bigram_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # unigram_counts[w] = total times w appeared as first word in a bigram
        self.unigram_counts: dict[str, int] = defaultdict(int)
        self.vocab: set[str] = set()
        self.is_trained: bool = False

    # ── Training ────────────────────────────────────────────────────────────

    def train(self, sentences: list[str]) -> None:
        """
        Train the model on a list of raw sentences.

        Parameters
        ----------
        sentences : list[str]
            Raw text sentences (will be cleaned internally).
        """
        all_tokens = []

        for sentence in sentences:
            tokens = self.cleaner.clean(sentence)

            all_tokens.extend(tokens)

            for i in range(len(tokens) - 1):
                w1, w2 = tokens[i], tokens[i + 1]

                self.bigram_counts[w1][w2] += 1
                self.unigram_counts[w1] += 1

        self.vocab = set(all_tokens)

        self.is_trained = True
        print(f"[BigramModel] Trained on {len(all_tokens)} tokens | "
              f"Vocab size: {len(self.vocab)} | "
              f"Unique bigrams: {sum(len(v) for v in self.bigram_counts.values())}")

    # ── Prediction ──────────────────────────────────────────────────────────

    def get_probability(self, word1: str, word2: str) -> float:
        """
        Return P(word2 | word1) using maximum likelihood estimation.

        P(w2|w1) = count(w1, w2) / count(w1)
        """
        word1 = word1.lower()
        word2 = word2.lower()
        total = self.unigram_counts.get(word1, 0)
        if total == 0:
            return 0.0
        return self.bigram_counts[word1].get(word2, 0) / total

    def suggest(self, last_word: str, n: int = 5) -> list[tuple[str, float]]:
        """
        Return the top-n most probable next words given the last word.

        Parameters
        ----------
        last_word : str
            The word the user just typed (last token in the input).
        n : int
            Number of suggestions to return.

        Returns
        -------
        list of (word, probability) tuples, sorted by probability descending.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet. Call .train() first.")

        word = last_word.lower()
        nexts = self.bigram_counts.get(word, {})

        if not nexts:
            return []

        total = self.unigram_counts[word]
        scored = [(w2, count / total) for w2, count in nexts.items()]
        scored.sort(key=lambda x: -x[1])

        return scored[:n]

    def suggest_from_sentence(self, sentence: str, n: int = 5) -> list[tuple[str, float]]:
        """
        Extract the last word from a sentence and return suggestions.

        Parameters
        ----------
        sentence : str
            Full sentence the user has typed so far.
        n : int
            Number of suggestions.
        """
        tokens = self.cleaner.clean(sentence)
        if not tokens:
            return []
        return self.suggest(tokens[-1], n)

    # ── Inspection helpers ──────────────────────────────────────────────────

    def get_bigram_table(self, top_n: int = 20) -> list[dict]:
        """
        Return a flat list of the most frequent bigram pairs for display.

        Returns
        -------
        list of dicts with keys: word1, word2, count, probability
        """
        rows = []
        for w1, nexts in self.bigram_counts.items():
            total = self.unigram_counts[w1]
            for w2, cnt in nexts.items():
                rows.append({
                    "word1": w1,
                    "word2": w2,
                    "count": cnt,
                    "probability": round(cnt / total, 4),
                })
        rows.sort(key=lambda r: -r["count"])
        return rows[:top_n]

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def get_unique_bigram_count(self) -> int:
        return sum(len(v) for v in self.bigram_counts.values())

