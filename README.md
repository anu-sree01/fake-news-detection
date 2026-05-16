# fake-news-detection

# Introduction

Fake news has become a major challenge in the digital era, influencing public opinion and spreading misinformation rapidly through social media and online platforms. This project focuses on detecting fake and real news articles using Natural Language Processing (NLP) and Machine Learning techniques.

The system analyzes textual content from news statements and predicts whether the news is reliable or misleading. The project uses the LIAR dataset along with TF-IDF vectorization and classification algorithms to perform fake news detection.

Project Objectives

- Detect fake and real news articles automatically

- Apply NLP preprocessing and feature extraction techniques

- Train machine learning classifiers for text classification

- Analyze linguistic patterns present in fake news

- Build a deployable web application using Streamlit

# Dataset Used

## LIAR Dataset

The project uses the LIAR dataset, a benchmark dataset for fake news detection.

The dataset contains:

- Short political statements
- Truthfulness labels
- Speaker information
- Contextual metadata

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
- 
## Machine Learning Models
- Logistic Regression
- Gradient Boosting (project workflow)
- Transformer/BERT concepts (project study)

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

The project trains a Logistic Regression classifier using TF-IDF vectorized news statements.

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


