# Food Hazard Detection from Recall Texts 
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/-HuggingFace-FDEE21?style=for-the-badge&logo=HuggingFace&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

> **Course:** NLP 053 - Computer Science and Engineering, University of Ioannina (CSE UOI) - 2026  
> **Challenge:** SemEval-2025 Task 9, Subtask 1 (ST1) | [Kaggle Competition](https://www.kaggle.com/t/e6a57812ae554144a90823a7ddf48fd0)

This repository contains the source code and experimental pipeline for our submission to **SemEval-2025 Task 9 (Subtask 1): Food Hazard Detection**. 

The goal of this project is the simultaneous classification of short food recall texts into specific Hazard and Product categories. To address the inherent challenges of the dataset—namely, severe long-tail data imbalance and strict error propagation in the official evaluation metric—we developed a **Weighted Soft-Voting Ensemble** architecture. 

Our final multimodal system dynamically combines the contextual understanding of fine-tuned Transformers (**BERT** and **RoBERTa**) with the statistical robustness of a calibrated classical classifier (**LinearSVC with TF-IDF**). The implementation utilizes custom dynamic class weighting integrated directly into the Cross-Entropy loss function to mitigate class imbalance.

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
├── csv/                     # Place train.csv, valid.csv, test.csv here (Also stores output submissions)
├── data/                    # DataProcessor and PyTorch FoodDataset classes
├── evaluation/              # Kaggle Scorer, Confusion Matrix Creator, Exporter, and PostProcessor
├── features/                # BoW, TF-IDF, and Word2Vec feature extraction wrappers
├── images/                  # Output directory for EDA plots, confusion matrices, and ablation studies
├── models/                  # Classifier wrappers, HF DualTrainer, saved model weights, and `.npy` probabilities
├── preprocessing/           # NLTK Text Cleaner and EDA Data Analyzer
├── trainer_utils/           # Automated pipeline runners for classical machine learning
├── utils/                   # Environment setup scripts (e.g., .env loaders for Hugging Face tokens)
├── 00_train_classical.ipynb # Baseline training, EDA, and exploration notebook
├── 01_train_bert.ipynb      # BERT fine-tuning, evaluation, and probability extraction
├── 02_train_roberta.ipynb   # RoBERTa fine-tuning, evaluation, and probability extraction
├── 03_multimodal_ensemble.ipynb # Fast SVM training and Weighted Soft-Voting Ensemble execution
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
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

### 5. Run the Code
Execute the provided Jupyter notebooks in the following order to reproduce the project's results:

* **`00_train_classical.ipynb`**: Run this locally (or in your preferred environment) to perform Exploratory Data Analysis (EDA), establish statistical baselines (Logistic Regression, Naive Bayes, SVM), and generate confusion matrices.
* **`01_train_bert.ipynb`**: Executes the fine-tuning of the `bert-base-uncased` model and saves its probability predictions.
* **`02_train_roberta.ipynb`**: Executes the fine-tuning of the `roberta-base` model and saves its probability predictions.
* **`03_multimodal_ensemble.ipynb`**: The final stage of the pipeline. It trains the calibrated SVM (on-the-fly), loads the Transformer probabilities from the disk, and evaluates the final **Weighted Soft-Voting Ensemble**, exporting the final `submission.csv` and the Ablation Study plot.

> **Important Note for Deep Learning Notebooks (`01`, `02`, and `03`):** > These specific notebooks are configured to run in **Google Colab** utilizing **Google Drive** mounting (`drive.mount('/content/drive')`). This is absolutely necessary to access sufficient GPU resources (e.g., NVIDIA T4) for training BERT and RoBERTa in a reasonable amount of time. 
> 
> **To run them:**
> 1. Upload the entire project folder to your Google Drive.
> 2. Open the notebooks via Google Colab.
> 3. Adjust the `PROJECT_PATH` variable in the first cell to match the exact path of the repository within your Drive.
