# 🧠 NLP Autofill System

An intelligent Autofill System built using **Natural Language Processing (NLP)** and a **Bigram Language Model** to predict the next most probable word based on user input.

The project demonstrates the fundamentals of NLP preprocessing, probabilistic language modeling, and real-time text prediction through an interactive Streamlit web application.

---

# 📌 Project Overview

This project solves the **Autofill Problem** using classical NLP techniques.

The system:
- Accepts user text input
- Cleans and preprocesses text
- Learns word relationships from a training corpus
- Predicts the most likely next words
- Displays ranked suggestions in real time

The prediction engine is powered by a **Bigram Language Model**, where the probability of the next word depends on the current word.

---

# ✨ Features

##  NLP Preprocessing Pipeline
- Lowercasing
- Punctuation removal
- Digit removal
- Tokenization
- Optional stopword filtering

---

##  Bigram Language Model
- Word-pair probability learning
- Maximum Likelihood Estimation (MLE)
- Ranked next-word prediction
- Top-K suggestions

---

##  Interactive Streamlit GUI
- Real-time autofill suggestions
- Dynamic probability visualization
- Bigram inspection table
- Data cleaning visualization
- Responsive UI design

---

##  Educational NLP Visualization
- Shows preprocessing steps
- Displays learned bigrams
- Demonstrates probability calculations
- Explains the NLP pipeline

---

# 🧠 How the Model Works

The system uses a **Bigram Language Model**.

For two consecutive words:

```text
P(w2 | w1) = count(w1, w2) / count(w1)
```

Where:
- `count(w1, w2)` → number of times the word pair appeared
- `count(w1)` → total occurrences of the first word

---

## Example

Given:

```text
the cat sat on the mat
```

The model learns:

```text
the → cat
cat → sat
sat → on
on → the
the → mat
```

If the user types:

```text
the
```

The model predicts:
- cat
- mat

based on learned probabilities.

---

# 🔄 NLP Pipeline

```text
Raw Text Corpus
     │
     ▼
Data Cleaning
(lowercase → remove punctuation → tokenize)
     │
     ▼
Bigram Training
(count word pairs)
     │
     ▼
Probability Calculation
P(w2|w1)
     │
     ▼
Prediction Engine
(rank next words)
     │
     ▼
Streamlit Interface
(display suggestions)
```

---

# 🖥️ Application Screens

## ✍️ Autofill Demo
Real-time next-word prediction system.

## 📊 Bigram Table
Displays the most frequent learned word pairs.

## 🧹 Data Cleaning
Visualizes preprocessing steps.

## 📖 How It Works
Explains the NLP architecture and probability logic.

---

# 📂 Project Structure

```text
NLP_AUTOFILL_PROJECT/
│
├── app.py
├── bigram_model.py
├── corpus.py
├── data_cleaning.py
├── requirements.txt
├── README.md
└──
```

---

# ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data display and tables |
| NumPy | Numerical operations |
| NLP Techniques | Text preprocessing |
| Bigram Model | Language prediction |

---


# 📊 Example Predictions

| User Input | Predicted Suggestions |
|---|---|
| the | quick, dog, cat |
| natural | language |
| machine | learning |
| we were | happy, excited |

---

# 🧪 Sample Training Corpus

The system is trained on a custom educational corpus containing:
- General English
- Technology
- Education
- Nature
- Common conversational phrases

---

# 📈 Future Improvements

- Trigram Language Model
- Transformer-based prediction
- User adaptive learning
- Large external datasets
- Sentence completion
- Multilingual support
- Speech-to-text integration

---

# 🎓 Academic Purpose

This project was developed as an academic NLP project to demonstrate:
- Text preprocessing
- Language modeling
- Probabilistic prediction
- Streamlit deployment
- Interactive NLP systems

---

# 👨‍💻 Author

**Ali Ahmed Zaki**



---
# 📜 License

This project is for educational and academic purposes.
