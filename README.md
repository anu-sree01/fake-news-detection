# FAKE-NEWS-DETECTION

# Introduction

Fake news has become a major challenge in the digital era, influencing public opinion and spreading misinformation rapidly through social media and online platforms. This project focuses on detecting fake and real news articles using Natural Language Processing (NLP) and Machine Learning techniques.

The system analyzes textual content from news statements and predicts whether the news is reliable or misleading. The project uses the LIAR dataset along with TF-IDF vectorization and classification algorithms to perform fake news detection.

Project Objectives

- Detect fake and real news articles automatically

- Apply NLP preprocessing and feature extraction techniques

- Train machine learning classifiers for text classification

- Analyze linguistic patterns present in fake news

- Build a deployable web application using Streamlit

# 👥 Team Members

| Name | Register No. |
| ------- | -------- | 
| Akshaya K P | 253214 |
| Anusree S | 253005 |


# Dataset Used

## LIAR Dataset

The project uses the LIAR dataset, a benchmark dataset for fake news detection.

The dataset contains:

- Short political statements
- Truthfulness labels
- Speaker information
- Contextual metadata

# Exploratory Data Analysis (EDA)

## Truthfulness Label Distribution

<img width="1527" height="616" alt="image" src="https://github.com/user-attachments/assets/0f788bab-b49f-45a0-93e1-0f9cd333caa1" />

## Binary label distribution (Fake vs Real)

<img width="704" height="584" alt="image" src="https://github.com/user-attachments/assets/bcd4e412-0c89-4b5e-8d39-9764a764201b" />

## Word cloud 

<img width="1904" height="656" alt="image" src="https://github.com/user-attachments/assets/fccce9e6-5ef9-438d-a280-8af40c6d1ef3" />

## Correlation Heatmap of Numeric Features 

<img width="1146" height="944" alt="image" src="https://github.com/user-attachments/assets/0ad5e27d-194f-4865-b271-5eb7a543dd65" />


# Technologies Used

## Programming Language

Python

## Libraries and Frameworks
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pickle

## NLP Techniques

- TF-IDF Vectorization
- Text preprocessing
- Feature extraction

## Machine Learning Models
-Logistic Regression (trained and deployed)

-Gradient Boosting (trained and evaluated)

-DistilBERT / Transformer (fine-tuned and evaluated)

# Project Workflow

1. Data Collection
2. Data Preprocessing
3. Text Cleaning
4. TF-IDF Feature Extraction
5. Model Training
6. Model Evaluation
7. Model Serialization using Pickle
8.Streamlit Deployment

# Model Training

## Models Used

### Logistic Regression
A linear machine learning model used for text classification and fake news detection using TF-IDF + hand-crafted features. Fast, interpretable, and strong on sparse text data. Powers the deployed Streamlit app.

### Gradient Boosting
An ensemble learning method included in the project workflow for improving classification performance.

### DistilBERT / Transformer Models
Advanced NLP transformer-based models studied for contextual understanding and fake news classification.

# Evaluation metrics
| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|---------:|----------:|-------:|---------:|--------:|
| Logistic Regression | 63-66% | ~0.64 |~0.63  | ~0.63 | ~0.68 |
| Gradient Boosting |65-68%  |~0.66  | ~0.65 | ~0.65 | ~0.70 |
| DistilBERT |68-72%  |~0.70  |~0.69  |~0.69  | ~0.75 |

**Best Model: DistilBERT**



## Steps Involved

- Load training and testing datasets
- Extract the statement column
- Convert text into numerical vectors using TF-IDF
- Train Logistic Regression classifier
- Evaluate accuracy
- Save trained model and vectorizer using Pickle
  
## Saved Files

- model.pkl
- vectorizer.pkl

# Streamlit Deployment

A web application was developed using Streamlit to allow users to test news statements interactively.


https://fake-news-detection-bl45vnkqyrck9c8qchtddr.streamlit.app/

## Features of the App

- User-friendly interface
- Text input for news statements
- Real-time prediction
- Fake/Real classification output
- Cached model loading for faster performance

# Sample Prediction Flow

1. User enters a news statement
2. Text is transformed using TF-IDF vectorizer
3. Trained model predicts the label
4. App displays:
    - Fake News
    - Real News
      
# Results

The model successfully performs fake news classification using NLP-based textual analysis.


