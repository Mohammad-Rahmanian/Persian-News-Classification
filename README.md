# Multi-Label Text Classification with LSTM and Machine Learning Models

## Overview
This project focuses on building and evaluating multiple models for multi-label text classification. It utilizes a dataset of documents that are categorized into various labels, implementing both traditional machine learning models and a Long Short-Term Memory (LSTM) network to handle sequence data effectively.

## Project Objectives
- **Loading Data and Categorizing Documents:** Organize documents based on directory structure into categories.
- **Display Dataset Information:** Provide statistics about the dataset including document counts and category distribution.
- **Text Preprocessing:** Normalize, tokenize, and lemmatize the text data, removing stopwords and punctuation.
- **Feature Extraction:** Use TF-IDF and Word2Vec for extracting text features.
- **Model Training and Evaluation:** Train and evaluate multiple machine learning models and an LSTM network.
- **Performance Visualization:** Visualize the performance of models through precision, recall, and confusion matrices.

## Installation and Setup
1. **Dependencies:**
   - Python 3.x
   - Libraries: `numpy`, `pandas`, `scikit-learn`, `TensorFlow`, `Keras`, `matplotlib`, `seaborn`, `gensim`, `hazm` (for Persian text)
2. **Dataset Setup:**
   - Store your dataset with a directory structure where each subfolder represents a category.
   - Set the `root_directory` in the script to the location of your dataset.

## Usage
Execute the script section by section:
1. **Data Loading and Preprocessing:** Load and categorize documents; display dataset statistics.
2. **Text Preprocessing and Feature Extraction:** Clean text data; extract features using TF-IDF and Word2Vec.
3. **Model Training and Evaluation:** Train and evaluate Naive Bayes, Random Forest, Logistic Regression, SVM, MLP, and LSTM models.
4. **Results Interpretation:** Compare model performances and analyze the impact of different feature extraction techniques.

## Models Included
- **Naive Bayes**
- **Random Forest**
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Multi-Layer Perceptron (MLP)**
- **Long Short-Term Memory Network (LSTM)**

## Conclusion
This project illustrates the application of various machine learning and deep learning models to multi-label text classification, providing insights into which models and feature extraction techniques perform best.

