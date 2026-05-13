import streamlit as st
import pandas as pd

from bigram_model import BigramModel
from corpus import get_corpus, get_corpus_stats


# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Autofill System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────────────────────
if "main_input" not in st.session_state:
    st.session_state.main_input = ""


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────
def set_example_text(text: str):
    st.session_state.main_input = text


def set_suggestion_text(text: str):
    st.session_state.main_input = text


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>

    /* Main Header */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }

    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1a1a2e;
        margin-bottom: 0.3rem;
    }

    .main-header p {
        font-size: 1.1rem;
        color: #666;
    }

    /* Suggestion Cards */
    .suggestion-card {
        background: #f8f9ff;
        border: 1px solid #dbe4ff;
        border-radius: 16px;
        padding: 1rem;
        margin: 8px 0;
        transition: all 0.25s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .suggestion-card:hover {
        background: #eef2ff;
        border-color: #4f6ef7;
        transform: translateY(-3px);
    }

    /* Better Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
        border: none;
        padding: 0.55rem;
    }

    /* Better Input */
    .stTextInput input {
        border-radius: 10px;
    }

    /* Footer */
    footer {
        visibility: hidden;
    }

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load & Cache Model
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(remove_stopwords=False):
    """
    Train and cache the Bigram model.
    Runs once unless settings change.
    """

    corpus = get_corpus()

    model = BigramModel(
        remove_stopwords=remove_stopwords
    )

    model.train(corpus)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Corpus Statistics
# ─────────────────────────────────────────────────────────────────────────────
corpus_stats = get_corpus_stats()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("## ⚙️ Settings")

    num_suggestions = st.slider(
        "Number of suggestions",
        min_value=3,
        max_value=10,
        value=6,
        step=1
    )

    remove_sw = st.checkbox(
        "Remove stopwords",
        value=False,
        help="Filter out common words like 'the', 'is', and 'and'."
    )

    if st.button("🔄 Retrain with new settings"):
        st.cache_resource.clear()
        st.rerun()

    model = load_model(remove_sw)

    st.divider()

    st.markdown("### 📊 Corpus Stats")

    st.metric("Sentences", corpus_stats["num_sentences"])
    st.metric("Total words", corpus_stats["total_words"])
    st.metric("Unique words", corpus_stats["unique_words"])
    st.metric("Unique bigrams", model.get_unique_bigram_count())

    st.divider()

    st.markdown("### ℹ️ About")

    st.markdown("""
    This system uses a **Bigram Language Model**
    to predict the next word based on the
    current word typed by the user.

    **Formula:**

    ```
    P(w2|w1) = count(w1,w2) / count(w1)
    ```
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Main Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 NLP Autofill System</h1>
    <p>
        Intelligent next-word prediction using
        Bigram Language Model
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "✍️ Autofill Demo",
    "📊 Bigram Table",
    "🧹 Data Cleaning",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Autofill Demo
# ─────────────────────────────────────────────────────────────────────────────
with tab1:

    st.subheader("Try the Autofill System")

    st.markdown("""
    Type a sentence below — suggestions appear
    based on the **last word** you typed.
    """)

    user_input = st.text_input(
        label="Your input:",
        placeholder="e.g. the quick",
        key="main_input",
        autocomplete="off",
    )

    if user_input.strip():

        suggestions = model.suggest_from_sentence(
            user_input,
            n=num_suggestions
        )

        if suggestions:

            tokens = user_input.strip().split()

            last_word = tokens[-1]

            prefix = " ".join(tokens[:-1])

            st.markdown(
                f"### Suggestions after `{last_word}`"
            )

            cols = st.columns(
                min(len(suggestions), 3)
            )

            for i, (word, prob) in enumerate(suggestions):

                full_sentence = (
                    prefix + " " + last_word + " " + word
                ).strip()

                with cols[i % 3]:

                    pct = round(prob * 100, 1)

                    st.markdown(
                        f"""
                        <div class="suggestion-card">

                            <div style="
                                font-size:18px;
                                font-weight:700;
                                margin-bottom:10px;
                            ">
                                🔵
                                <span style="color:#1f3c88">
                                    {last_word}
                                </span>

                                <span style="color:#222">
                                    {word}
                                </span>
                            </div>

                            <div style="
                                font-size:13px;
                                color:#666;
                                margin-bottom:8px;
                            ">
                                Probability: {pct}%
                            </div>

                            <div style="
                                width:100%;
                                height:8px;
                                background:#dbe4ff;
                                border-radius:999px;
                                overflow:hidden;
                            ">

                                <div style="
                                    width:{pct}%;
                                    height:100%;
                                    background:linear-gradient(
                                        90deg,
                                        #4f6ef7,
                                        #7c9cff
                                    );
                                    border-radius:999px;
                                ">
                                </div>

                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.divider()

            st.markdown("### Click to use a suggestion")

            for word, prob in suggestions:

                full_sentence = (
                    prefix + " " + last_word + " " + word
                ).strip()

                st.button(
                    f"→ {full_sentence}",
                    key=f"btn_{word}",
                    on_click=set_suggestion_text,
                    args=(full_sentence + " ",)
                )

        else:

            st.warning(
                f"""
                No suggestions found for the word
                '{user_input.strip().split()[-1]}'.

                Try another word or add more
                sentences to the corpus.
                """
            )

    else:

        st.info(
            "Start typing to see autofill suggestions..."
        )

        st.markdown("### Quick Examples")

        examples = [
            "the quick",
            "natural language",
            "we were",
            "machine learning",
            "the dog"
        ]

        ex_cols = st.columns(len(examples))

        for i, ex in enumerate(examples):

            with ex_cols[i]:

                st.button(
                    ex,
                    key=f"ex_{i}",
                    on_click=set_example_text,
                    args=(ex,)
                )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Bigram Table
# ─────────────────────────────────────────────────────────────────────────────
with tab2:

    st.subheader("Bigram Probability Table")

    st.markdown("""
    Top bigram pairs learned from the
    training corpus.
    """)

    top_n = st.slider(
        "Show top N bigrams",
        10,
        50,
        20,
        5
    )

    rows = model.get_bigram_table(top_n)

    search_word = st.text_input(
        "Filter by first word:",
        placeholder="e.g. the"
    )

    if search_word.strip():

        rows = [
            r for r in rows
            if r["word1"] == search_word.strip().lower()
        ]

    if rows:

        df = pd.DataFrame(rows)

        df.columns = [
            "Word 1",
            "Word 2",
            "Count",
            "Probability"
        ]

        df["Probability (%)"] = (
            df["Probability"] * 100
        ).round(1).astype(str) + "%"

        st.dataframe(
            df[
                [
                    "Word 1",
                    "Word 2",
                    "Count",
                    "Probability (%)"
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(
            df.set_index("Word 2")["Count"].head(15),
            use_container_width=True,
        )

    else:

        st.warning(
            "No bigrams found for that word."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Data Cleaning
# ─────────────────────────────────────────────────────────────────────────────
with tab3:

    st.subheader("Data Cleaning Pipeline")

    st.markdown("""
    See exactly how raw text is transformed
    before training.
    """)

    raw_text = st.text_area(
        "Enter raw text to clean:",
        value="Hello, World! This is an NLP Project — version 2.0.",
        height=120,
    )

    if raw_text.strip():

        from data_cleaning import DataCleaning

        cleaner = DataCleaning()

        step1 = cleaner.to_lowercase(raw_text)

        step2 = cleaner.remove_punctuation(step1)

        tokens = cleaner.tokenize(step2)

        col_a, col_b = st.columns(2)

        with col_a:

            st.markdown("### Step 1 — Lowercase")
            st.code(step1)

            st.markdown(
                "### Step 2 — Remove Punctuation & Digits"
            )

            st.code(step2)

        with col_b:

            st.markdown("### Step 3 — Tokenization")
            st.code(str(tokens))

            st.markdown("### Step 4 — Token Count")

            st.success(
                f"{len(tokens)} tokens extracted"
            )

        st.markdown("### Token Visualization")

        token_html = " ".join(
            f'''
            <span style="
                background:#e8ecff;
                border:1px solid #c0caff;
                border-radius:6px;
                padding:4px 10px;
                margin:3px;
                display:inline-block;
                font-family:monospace;
                font-size:14px;
            ">
                {t}
            </span>
            '''
            for t in tokens
        )

        st.markdown(
            token_html,
            unsafe_allow_html=True
        )
