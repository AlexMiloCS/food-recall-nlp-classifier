# Food Hazard Detection from Recall Texts 

> **Course:** NLP 053 - Computer Science and Engineering, University of Ioannina (CSE UOI) - 2026
> **Challenge:** SemEval-2025 Task 9, Subtask 1 (ST1) | [Kaggle Competition](https://www.kaggle.com/t/e6a57812ae554144a90823a7ddf48fd0)

## Overview
This repository contains the code, data analysis, and final pipeline for the **Food Hazard Prediction** task. The objective is to build explainable classification systems capable of predicting hazard and product categories from the titles and texts of food-incident recall reports collected from the web.

The system predicts two coarse-grained labels per text:
* **Hazard Category:** 10 distinct classes (e.g., Allergens, Biological).
* **Product Category:** 22 distinct classes.

## Dataset & Challenges
The dataset consists of English recall titles and full texts, heavily characterized by a **long-tail / imbalanced label distribution**, making this a challenging real-world NLP task. 

* **Train:** 5,082 instances
* **Validation:** 565 instances
* **Test:** 997 instances
* **Total Challenge Data:** 6,644 English instances.

*Note: The official data files are not tracked in this repository due to size and licensing constraints.*

## Evaluation Metric
The official scoring metric is not a simple average. It strictly prioritizes the correct prediction of the hazard category to mitigate error propagation. The final score is computed as:

$$\text{Official Score} = \frac{\text{Macro-F1}_{hazard} + \text{Macro-F1}_{product | hazard\_correct}}{2}$$

*If the hazard prediction is incorrect, the product prediction does not contribute to the score for that specific example.*

## Methodology
To tackle this multi-class text categorization problem, we implemented a progressive, Object-Oriented experimental pipeline:

1. **Exploratory Data Analysis (EDA) & Preprocessing:**
   * Text cleaning, tokenization, and normalization (Lemmatization, Stop-word removal).
   * Dynamic calculation of class weights to handle the severe long-tail distribution.
2. **Classical Baselines:**
   * Feature extraction: BoW, TF-IDF, and Word2Vec.
   * Classifiers: LinearSVC (SVM), Logistic Regression, Naive Bayes.
   * Extensive Hyperparameter tuning.
3. **Deep Learning (Transformers):**
   * Fine-tuning state-of-the-art contextual embeddings (`bert-base-uncased` and `roberta-base`).
   * Custom Trainer implementation to integrate class weights directly into the CrossEntropyLoss.
4. **Multimodal Ensemble (Final System):**
   * A Weighted Soft-Voting Ensemble combining the probabilistic outputs of BERT (35%), RoBERTa (35%), and a Calibrated SVM (30% via Platt Scaling).
5. **Evaluation & Error Analysis:**
   * Automated generation of Confusion Matrices and Ablation Studies to understand model behavior on minority classes.

## Repository Structure
```text
├── csv/                    # (Ignored) Place train.csv, valid.csv, and test.csv here
├── data/                   # Dataset classes and PyTorch tensor processors
├── preprocessing/          # Text cleaning and EDA statistical scripts
├── features/               # BoW, TF-IDF, and Word2Vec extractors
├── models/                 # Classical classifier wrappers and Custom Neural Trainers
├── evaluation/             # ST1 metric scorers and automated Confusion Matrix generators
├── images/                 # Output directory for generated plots and matrices
├── multimodal_ensemble.ipynb # Final notebook for Ensemble evaluation and submission
├── train_classical.ipynb   # Baseline training and exploration notebook
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
