## Yelp Review Rating Classification

A machine learning project for sentiment analysis and rating prediction of Yelp reviews using TF-IDF feature extraction and neural networks.

### Project Overview

This project implements a multi-class classification system to predict Yelp review ratings (1-5 stars) based on review text. The solution combines traditional NLP feature extraction methods with neural network classifiers.

### Dataset

We use the **Yelp Review Full** dataset from Hugging Face:
- **Dataset**: [Yelp/yelp_review_full](https://huggingface.co/datasets/Yelp/yelp_review_full)
- **Features**:
  - `text`: The review text content
  - `label`: Rating label (0-4, corresponding to 1-5 stars respectively)

### Technical Approach

#### Data Preprocessing
- Split the original training data to create a dedicated validation set
- Applied dataset size constraints (training set limited to 100,000 samples)
- Maintained proportional splits for validation and test sets

#### Feature Extraction
- **TF-IDF (Term Frequency-Inverse Document Frequency)** implementation
- **Bag-of-Words** representation using `TfidfVectorizer` from `scikit-learn`
- Text vectorization for neural network compatibility

