# NLP--Twitter-Sentiment-Classifier

Three-class sentiment classifier built on a dataset of ~163k tweets. Labels: negative, neutral, positive. Evaluated on accuracy and macro-F1.

## Task

Given a tweet's text, predict whether the sentiment is negative (−1), neutral (0), or positive (+1). The dataset comes from Indian political Twitter and is moderately imbalanced — positive tweets make up roughly 44% of the data.

## Models

**Baseline — TF-IDF + Multinomial Naive Bayes**
Unigram and bigram features (60k max), sublinear TF scaling, alpha=0.1. Trains in seconds, sets the floor for comparison.

**Neural — Embedding + Dual-Kernel 1D CNN**
Learned embedding (vocab=40k, dim=128) feeding two parallel Conv1D branches with kernel sizes 3 and 5. Outputs are concatenated, passed through dropout and dense layers, then softmax over three classes. Class weights applied to handle imbalance. Trained with Adam, categorical cross-entropy, early stopping on val loss.

## Results

| Model | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + Naive Bayes | 75.09% | 0.7289 |
| Embedding + CNN | 96.33% | 0.9595 |

The CNN improved over the baseline by ~21 points on accuracy. The biggest gain was on the neutral class (F1: 0.762 → 0.984), which Naive Bayes consistently confused with positive due to class imbalance.

## Dataset

`Twitter_Data.xlsx` — two columns: `clean_text` and `category`. Upload to your Colab/Kaggle session before running.

The notebook reads the file directly from `.xlsx` — no conversion needed.

## How to Run

**On Google Colab:**
1. Upload `Twitter_Data.xlsx` to the session
2. Open the notebook and run all cells top to bottom
3. Training takes roughly 10–20 minutes on CPU (faster with GPU)

**On Kaggle:**
1. Upload `Twitter_Data.xlsx` as a dataset
2. Update `DATA_PATH` in the loading cell to match your dataset path
3. Enable GPU (T4) for faster training
4. Run all cells

## Files

```
├── CS5143_NLP_PA1_Sentiment_Classifier.ipynb   # main notebook
├── Twitter_Data.xlsx                            # dataset
└── PA1_Report_Moez_K247840.pdf                 # assignment report
```

## Environment

- Python 3.10+
- TensorFlow 2.x
- scikit-learn
- pandas, numpy, matplotlib, seaborn, openpyxl

Install: `pip install tensorflow scikit-learn pandas numpy matplotlib seaborn openpyxl`

## Key Findings

The CNN handles positive and negative tweets well but the remaining errors cluster around three things: sarcasm (positive surface words masking criticism), mixed-sentiment tweets where one clause contradicts another, and trailing negation that falls outside a short convolutional window. A BiLSTM or transformer would recover some of these cases at the cost of longer training time.

---
**Course:** CS5143 — Natural Language Processing, Spring 2026  
**Student:** Moez Ur Rehman 
**Institution:** FAST-NUCES
