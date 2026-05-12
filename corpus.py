"""
Corpus Module
=============
Contains the training text data used by the Bigram model.
You can extend CORPUS_SENTENCES with more sentences to improve suggestions.
"""

CORPUS_SENTENCES = [
    # General English
    "the quick brown fox jumps over the lazy dog",
    "the cat sat on the mat and the cat was happy",
    "natural language processing is amazing and fun",
    "the big brown bear walked in the forest slowly",
    "machine learning is a subset of artificial intelligence",
    "the quick cat ran over the lazy fox today",
    "language models predict the next word in sequence",
    "deep learning and natural language processing are related",
    "the dog ran fast across the green field",
    "we were happy and we were excited about the project",
    "we love natural language processing and machine learning",
    "the model predicts words based on previous context",
    "artificial intelligence is changing the world rapidly",
    "we hated waiting but we loved the results",

    # Technology
    "python is a popular programming language for data science",
    "data science combines statistics and machine learning",
    "neural networks learn from large amounts of data",
    "the internet has transformed how we communicate and learn",
    "software engineers build systems that solve real problems",
    "programming languages like python and java are widely used",
    "computers process information much faster than humans",
    "algorithms are the foundation of all software applications",

    # Nature
    "the sun rises in the east and sets in the west",
    "trees grow slowly but they live for hundreds of years",
    "water is essential for all forms of life on earth",
    "animals adapt to their environments over thousands of years",
    "the ocean covers more than seventy percent of the earth",

    # Education
    "students learn better when they practice regularly",
    "reading books improves vocabulary and critical thinking",
    "education is the key to a successful and happy life",
    "universities provide opportunities for research and discovery",
    "learning new skills helps people grow professionally",

    # Common phrases
    "in the beginning there was a lot of uncertainty",
    "at the end of the day we must make a decision",
    "the best way to learn is to practice every day",
    "we need to work together to achieve our goals",
    "the project was completed on time and within budget",
    "we are making progress on the natural language model",
    "this system uses bigram probabilities to predict words",
]


def get_corpus() -> list[str]:
    """Return the full corpus as a list of sentences."""
    return CORPUS_SENTENCES


def get_corpus_stats() -> dict:
    """Return basic statistics about the corpus."""
    words = " ".join(CORPUS_SENTENCES).lower().split()
    return {
        "num_sentences": len(CORPUS_SENTENCES),
        "total_words": len(words),
        "unique_words": len(set(words)),
    }

