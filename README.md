# A Comparative Study of CNN Architectures for Histopathological Image Classification 🔬💻

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18+-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: KMC RCC](https://img.shields.io/badge/Dataset-KMC_RCC-blue)](https://www.kaggle.com/datasets/shreyan983/kmc-renal)

This repository contains the official code, models, and evaluation results for the paper: **"A Comparative Study of CNN Architectures for Histopathological Image Classification."** 

This study provides a unified comparative analysis of five state-of-the-art pretrained Convolutional Neural Networks (CNNs) for the automated, multi-class grading of Renal Cell Carcinoma (RCC) using the publicly available KMC RCC dataset.

---

## Table of Contents
1. [Overview & Methodology](#-overview--methodology)
2. [Dataset](#dataset)
3.[Repository Structure](#-repository-structure)
4. [File Descriptions](#-file-descriptions)
5.[Key Results](#-key-results)
6. [How to Run](#-how-to-run)
7. [Citation](#-citation)
8. [Authors](#-authors)

---

## Overview & Methodology

Automated grading of renal cell carcinoma from histopathological images is essential for prognosis and treatment planning. This repository implements a robust **two-phase transfer learning strategy** across five diverse CNN backbones:
1. **DenseNet121** (Heavyweight / Dense connections)
2. **InceptionResNetV2** (Heavyweight / Hybrid connections)
3. **InceptionV3** (Heavyweight / Inception modules)
4. **MobileNetV3-Small** (Lightweight / Mobile-optimized)
5. **EfficientNetB0** (Lightweight / Compound scaling)

### Unified Training Strategy
To ensure a strictly fair comparison, all models share:
* Identical training conditions and data augmentation pipelines.
* A consistent, lightweight classification head (Global Average Pooling + 2 Dense Layers with Dropout).
* **Phase 1 (Frozen Base):** Training the custom head for 10 epochs while the CNN backbone is frozen.
* **Phase 2 (Fine-tuning):** Unfreezing the top 30% of the backbone and fine-tuning with a reduced learning rate for up to 50 epochs with early stopping.

---

## Dataset

Experiments were conducted on the **KMC Renal Cell Carcinoma (RCC) Dataset**, which contains 4,013 H&E-stained images categorized into 5 tumor grades (Grade 0 to Grade 4).
* **Link:** [KMC-RCC Dataset on Kaggle](https://www.kaggle.com/datasets/shreyan983/kmc-renal)
* **Splits:** 80% Train, 10% Validation, 10% Test (Stratified). The training set was balanced via random oversampling.

---

## Repository Structure

The repository is organized into five main directories, one for each evaluated architecture. Every directory is self-contained with its specific training script, saved models, training logs, and generated visual plots.

├── DenseNet121/
├── EfficientNetB0/
├── InceptionResNetV2/
├── InceptionV3/
└── MobileNetV3Small/
Inside each architecture folder:
code
Text[Architecture_Name]/
│
├── [Architecture_Name].py                 # The main Python training/evaluation script
├── best_model_phase1.keras                # Checkpoint: Best weights from initial frozen training
├── best_model_final.keras                 # Checkpoint: Best weights after fine-tuning
├── final_model.keras                      # The final exported model
│
├── evaluation_results.json                # Comprehensive metrics (F1, Precision, Recall, AUC, etc.)
├── experiment_summary.json                # Metadata (hyperparameters, splits, final accuracies)
│
├── Log.txt                                # Raw console outputs during execution
├── training_phase1.csv                    # Epoch-by-epoch logs for Phase 1
├── training_phase2.csv                    # Epoch-by-epoch logs for Phase 2
│
├── fig1_training_history.png              # Accuracy/Loss curves across both phases
├── fig2_confusion_matrix.png              # Normalized & Raw confusion matrices
├── fig3_roc_curves.png                    # One-vs-Rest ROC curves per class
├── fig4_precision_recall_curves.png       # Precision-Recall curves per class
├── fig5_per_class_metrics.png             # Bar chart of precision, recall, and F1 per class
├── fig6_class_distribution.png            # Data distribution across Train/Val/Test splits
└── training_history.png                   # Summarized training visualization
Key Results
Our findings establish that deeper, higher-capacity models significantly outperform lightweight models for fine-grained histopathological grading when using standard fine-tuning strategies. DenseNet121 emerged as the optimal backbone.
Model	Accuracy	Macro-F1	Precision	Recall	Cohen's κ
DenseNet121	93.78%	93.63%	93.69%	93.67%	92.22%
InceptionResNetV2	93.28%	93.11%	93.25%	93.10%	91.59%
InceptionV3	91.79%	91.53%	91.80%	91.59%	89.72%
MobileNetV3-Small	73.13%	73.39%	75.20%	73.62%	66.46%
EfficientNetB0	47.26%	43.17%	45.89%	47.19%	33.89%
Detailed per-class metrics and confusion matrices can be found in the respective architecture folders.
How to Run
1. Prerequisites
Ensure you have Python 3.8+ installed along with the required libraries:
!pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
2. Dataset Setup
Download the dataset from Kaggle and update the DATA_PATH variable inside the .py script of the model you wish to run to point to your local dataset directory.
3. Execution
Navigate to the desired model's folder and execute the script. For example, to train DenseNet121:
cd DenseNet121
python DenseNet121.py
Note: The scripts are configured to automatically utilize available GPUs and mixed precision (float16) for accelerated training.
Citation
If you use this code, the methodology, or the pre-trained models in your research, please cite our paper:
Bibtex
@inproceedings{...,
  title={A Comparative Study of CNN Architectures for Histopathological Image Classification},
  author={...},
  booktitle={...},
  year={...},
  organization={...}
}

For questions or collaborations, please reach out via the provided university email addresses in the paper.
