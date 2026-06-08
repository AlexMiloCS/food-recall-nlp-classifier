# Food Hazard Detection from Recall Texts 

> **Course:** NLP 053 - Computer Science and Engineering, University of Ioannina (CSE UOI) - 2026
> **Challenge:** SemEval-2025 Task 9, Subtask 1 (ST1) | [Kaggle Competition](https://www.kaggle.com/t/e6a57812ae554144a90823a7ddf48fd0)

## Overview
This repository contains the code, data analysis, and final report for the **Food Hazard Prediction** task. The objective is to build explainable classification systems capable of predicting hazard and product categories from the titles and texts of food-incident recall reports collected from the web.

The model predicts two coarse-grained labels per text:
* **Hazard Category:** 10 distinct classes.
* **Product Category:** 22 distinct classes.

## Dataset & Challenges
The dataset consists of English recall titles and full texts, heavily characterized by a **long-tail / imbalanced label distribution**, making this a challenging real-world NLP task. 

* **Train:** 5,082 instances
* **Validation:** 565 instances
* **Test:** 997 instances
* **Total Challenge Data:** 6,644 English instances.

*Note: The broader Food Recall Incidents dataset (Zenodo, CC BY-NC-SA 4.0) contains additional multilingual data.*

## Evaluation Metric
The official scoring metric is not a simple average. It strictly prioritizes the correct prediction of the hazard category to mitigate error propagation. The final score is computed as:

$$ \text{Official Score} = \frac{\text{Macro-F1}_{hazard} + \text{Macro-F1}_{product | hazard\_correct}}{2} $$

*If the hazard prediction is incorrect, the product prediction does not contribute to the score for that specific example.*

## Methodology
To tackle this multi-class text categorization problem, we implemented a progressive experimental pipeline:

1. **Exploratory Data Analysis (EDA) & Preprocessing:**
   * Handled class imbalance using strategic sampling and class weights.
   * Text cleaning, tokenization, and normalization using `spacy` and `nltk`.
2. **Classical Baselines:**
   * Feature extraction: TF-IDF, Word2Vec, GloVe, and FastText (`gensim`).
   * Classifiers: Logistic Regression, Random Forest, SVM (`scikit-learn`).
3. **Neural Baselines:**
   * Feedforward Neural Networks and basic RNN/LSTM architectures to capture sequential context.
4. **State-of-the-Art (SOTA) Approaches:**
   * Fine-tuning Transformer-based embeddings (e.g., BERT, RoBERTa) tailored for heavily imbalanced text classification.
5. **Evaluation & Error Analysis:**
   * Thorough cross-validation setup.
   * Ablation studies, confusion matrix analysis, and class-wise performance discussion to understand failure modes on the long-tail classes.

## Repository Structure
```text
├── data/                   # Raw and preprocessed dataset files (not tracked if large)
├── notebooks/              # Jupyter notebooks for EDA, baselines, and SOTA experiments
├── models/                 # Saved model weights and tokenizers
├── reports/                # Final PDF report and presentation slides
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```
## Installation & Usage

1. Clone the repository:
```bash
   git clone [https://github.com/yourusername/food-hazard-detection.git](https://github.com/yourusername/food-hazard-detection.git)
   cd food-hazard-detection
