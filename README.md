<div style="display: flex; align-items: center;">
  <h1>Multi-Label Text Classification with LSTM and Machine Learning Models</h1>
  <img src="https://github.com/Mohammad-Rahmanian/Persian-News-Classification/assets/78559411/6560b2f8-76aa-4836-aefe-4d46c74e2bfe" alt="News" width="100">
</div>

## Overview  📌
This project focuses on building and evaluating multiple models for multi-label text classification of Persian news articles. It utilizes a diverse dataset of documents categorized into various labels, such as politics, social issues, and culture, implementing both traditional machine learning models and a Long Short-Term Memory (LSTM) network to handle sequence data effectively.

## Dataset Description
The dataset consists of a comprehensive collection of Persian news articles, organized into a hierarchical structure of categories and subcategories.

### Downloading the Dataset 
<div style="display: flex; align-items: center;">
  <img src="https://github.com/Mohammad-Rahmanian/Persian-News-Classification/assets/78559411/56ef29cb-c84b-4206-bd1f-b104517f89a7" alt="Dataset" width="50">
</div>
The dataset can be accessed and downloaded from the following Google Drive link:

[Download Dataset](https://drive.google.com/drive/u/5/folders/1kA2gcSPwF3jLIgjffY-zeLrlbh0Pih75)


**Steps to download and set up the dataset:**
1. Click on the link to navigate to Google Drive.
2. Download the required dataset files.
3. Place the downloaded files into the root directory of the project, ensuring they are named correctly as per the scripts' configuration.

## Project Objectives 🌟
- **Loading Data and Categorizing Documents:** Organize documents based on directory structure into categories.
- **Display Dataset Information:** Provide statistics about the dataset including document counts and category distribution.
- **Text Preprocessing:** Normalize, tokenize, and lemmatize the text data, removing stopwords and punctuation.
- **Identify Key Terms with TF-IDF:** Apply TF-IDF vectorization to highlight key terms that characterize each category.
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

## Usage 📘
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
