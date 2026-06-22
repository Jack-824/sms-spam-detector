# sms-spam-detector
Machine learning project for SMS spam detection using SVM
9. Spam Detection using SVM
This project is about building an SMS spam detector using a Support Vector Machine.
The goal is to identify whether a message is spam or legitimate based on its text content.

## Overview
This project builds and evaluates a machine-learning classifier that labels SMS
messages as **spam** or **ham** (legitimate). It compares two text-to-number
conversion methods (CountVectorizer and TF-IDF) feeding a linear Support Vector
Machine, and investigates how much text **preprocessing** affects the result.

## Dataset
- `spam.csv` — the SMS Spam Collection (~5,572 messages), Latin-1 encoded.
- Columns used: `v1` (label: ham/spam) and `v2` (message text).
- Class balance: ~87% ham, ~13% spam — **imbalanced**, which shapes how we
  measure success.

## Requirements
- Python 3.10 (conda env `inm`)
- `pandas`, `numpy`, `scikit-learn`, `nltk`, `matplotlib`, `seaborn`
- First run downloads NLTK stopwords automatically:
  `nltk.download('stopwords', quiet=True)`

## How to Run
- Place `spam.csv` in the same folder as the notebook.
- Open the notebook and run **Kernel → Restart & Run All** (cells must run in
   order, since later cells depend on earlier variables).

---

## Pipeline Walkthrough

### Phase 1 — Data Acquisition & Preprocessing
- **Load & clean**: drop empty columns, rename to `label`/`Text`, and encode
  labels numerically (`ham`→0, `spam`→1) in `label_enc`.
- **EDA**: a pie chart shows the class imbalance; a message-length histogram
  shows spam messages are systematically longer than ham.
- **Text cleaning** (`clean_text`): for each message —
  1. remove non-alphabetic characters (regex),
  2. lowercase,
  3. remove **stopwords** (common words like *the*, *and*),
  4. apply **stemming** (PorterStemmer reduces words to roots, e.g.
     *winning → win*).
- **Train/test split**: 80/20, `stratify`ed on the label so both sets keep the
  same spam ratio. `random_state=42` makes it reproducible.

### Phase 2 — Vectorization (text → numbers)
SVMs need numeric input, so each message becomes a vector over the vocabulary.
- **CountVectorizer**: each feature = how many times a word appears.
- **TF-IDF**: word counts re-weighted so common words shrink and rare,
  informative words grow.
- **No leakage**: the vectorizer is `fit` on the **training set only**, then used
  to `transform` both sets.

### Phase 3 — SVM Training
Two linear SVMs (`SVC(kernel='linear')`) are trained — one on Count features,
one on TF-IDF features — then used to predict on the test set.

### Phase 4 — Evaluation
- **Confusion matrix** + **classification report** (precision, recall, F1) for
  each model, shown as numbers and as heatmaps.
- Because the data is imbalanced, **spam recall** (how much spam we catch) and
  **F1** matter more than raw accuracy.

### Phase 5 — Role of Preprocessing (Raw vs Preprocessed)
A controlled 2×2 experiment runs both vectorizers on **raw** and **cleaned**
text, holding everything else constant, to isolate what TF-IDF actually adds.

---

## Key Concepts

| Concept | One-line meaning |
|---|---|
| **Stopwords** | Frequent low-information words removed before modeling. |
| **Stemming** | Reducing words to a common root so variants count as one feature. |
| **CountVectorizer** | Represents text by raw word counts. |
| **TF-IDF** | Word counts down-weighted by how common a word is across all messages. |
| **SVM** | Finds the maximum-margin hyperplane separating spam from ham. |
| **Precision** | Of messages flagged spam, how many really were spam. |
| **Recall** | Of all real spam, how much was caught. |
| **F1** | Harmonic mean of precision and recall. |
| **Class imbalance** | One class dominates, so accuracy alone is misleading. |
| **Data leakage** | Letting test info influence training; avoided by fitting the vectorizer on train only. |

---

## Results & Findings
- All models reach ~98.6% accuracy.
- On **preprocessed** text, Count and TF-IDF are effectively **tied** (difference
  of one message — within noise).
- On **raw** text, TF-IDF clearly beats Count (recall 0.89 vs 0.87), because IDF
  down-weights the common words that cleaning would otherwise remove.
- **Takeaway**: preprocessing was a bigger lever than the vectorizer choice.
  Final model: **SVM + CountVectorizer on preprocessed text** — equally accurate,
  simpler, and strongest on spam recall.

## Limitations & Future Work
- Single train/test split; cross-validation would give confidence intervals.
- Small, imbalanced dataset; class weighting or more data could lift recall.
- Future: tune the SVM `C` parameter, try n-grams, add a Naïve Bayes baseline.
