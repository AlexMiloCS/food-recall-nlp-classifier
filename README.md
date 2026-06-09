# Food Hazard Detection from Recall Texts 

> **Course:** NLP 053 - Computer Science and Engineering, University of Ioannina (CSE UOI) - 2026  
> **Challenge:** SemEval-2025 Task 9, Subtask 1 (ST1) | [Kaggle Competition](https://www.kaggle.com/t/e6a57812ae554144a90823a7ddf48fd0)

## Overview
This repository contains the source code, Object-Oriented framework, and final pipeline for the **Food Hazard Prediction** task. The objective is to build explainable classification systems capable of predicting hazard and product categories from the titles and texts of food-incident recall reports.

The system predicts two coarse-grained labels per text:
* **Hazard Category:** 10 distinct classes.
* **Product Category:** 22 distinct classes.

## Dataset & Challenges
The dataset consists of English recall titles and full texts, heavily characterized by a **long-tail / imbalanced label distribution**. 

* **Train:** 5,082 instances
* **Validation:** 565 instances
* **Test:** 997 instances

*Note: The official data files (`train.csv`, `valid.csv`, `test.csv`) are not tracked in this repository due to size and licensing constraints. Please place them in the `csv/` directory before running the code.*

## Methodology & Architecture
We implemented a progressive, modular Object-Oriented pipeline:

1. **Preprocessing (`preprocessing/`, `data/`):** Text cleaning (lemmatization, stop-word removal via NLTK) and dynamic Class Weight calculation to combat imbalance.
2. **Classical Baselines (`features/`, `models/`):** Feature extraction via custom BoW, TF-IDF, and Word2Vec wrappers. Classification using tuned SVM, Logistic Regression, and Naive Bayes.
3. **Deep Learning:** Fine-tuning state-of-the-art contextual embeddings (`bert-base-uncased` and `roberta-base`) utilizing a custom Hugging Face Trainer to inject class weights into the Loss Function.
4. **Multimodal Ensemble:** A Weighted Soft-Voting system combining probabilistic outputs: BERT (35%), RoBERTa (35%), and Calibrated SVM (30% via Platt Scaling).

## Directory Structure
```text
├── csv/                    # (Ignored) Place train.csv, valid.csv, and test.csv here
├── data/                   # DataProcessor and PyTorch FoodDataset classes
├── evaluation/             # ST1 Kaggle Scorer, Confusion Matrix Creator, and Exporter
├── features/               # BoW, TF-IDF, and Word2Vec feature extraction wrappers
├── images/                 # Output directory for generated EDA plots and matrices
├── models/                 # Classifier wrappers (SVM, Logistic, NB) and Custom HF Trainer
├── preprocessing/          # NLTK Text Cleaner and EDA Data Analyzer
├── trainer_utils/          # Automated pipeline runners (ModelTrainer, ClassicalTrainer)
├── utils/                  # Environment setup scripts (.env loaders)
├── train_classical.ipynb   # Baseline training, EDA, and exploration notebook
├── multimodal_ensemble.ipynb # Deep Learning fine-tuning and final Ensemble creation
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Installation & Usage

1. **Clone the repository and navigate into it:**
```bash
git clone https://github.com/yourusername/food-hazard-detection.git
cd food-hazard-detection
```
*(Note: Remember to replace `yourusername` in the URL above with your actual GitHub username).*

2. **Install the required dependencies:**
```bash
pip install -r requirements.txt
```

3. **Environment Setup:** Create a `.env` file in the root directory and add your Hugging Face Token (required for downloading Transformer models):
```text
HF_TOKEN=your_huggingface_token_here
```

4. **Data Setup:** Download the official dataset from Kaggle and place the CSV files (`train.csv`, `valid.csv`, `test.csv`) inside the `csv/` directory.

5. **Run the Code:** 
   * Run `train_classical.ipynb` locally (or in your preferred environment) to establish baselines and generate EDA plots.
   * Run `multimodal_ensemble.ipynb` to execute the deep learning models and final ensemble evaluation. **Important Note:** This specific notebook is configured to run in **Google Colab** utilizing **Google Drive** mounting (`drive.mount('/content/drive')`). This is necessary to access sufficient GPU resources for training BERT and RoBERTa. To run it, upload the repository folder to your Google Drive and adjust the `PROJECT_PATH` variable in the notebook accordingly.
