# Text Classification Project

## Overview

This project aims to classify text data into five categories using different text embedding techniques and machine learning models. The dataset contains text samples and corresponding labels. The project includes data exploration, preprocessing, and evaluation of three different text embedding methods: Word2Vec, GloVe, and BERT. Each method's performance is compared using Logistic Regression.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Exploration](#data-exploration)
3. [Text Preprocessing](#text-preprocessing)
4. [Text Embedding Techniques](#text-embedding-techniques)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Results](#results)
7. [Usage](#usage)
8. [Dependencies](#dependencies)

## Introduction

This project uses a dataset with text samples and corresponding labels. The goal is to build and evaluate text classification models using different embedding techniques:

- **Word2Vec**: A word embedding model that represents words in vector space.
- **GloVe**: A pre-trained word embedding model.
- **BERT**: A contextualized word embedding model that captures the meaning of words based on their context.

## Data Exploration

The dataset contains columns:

- **Text**: The text content.
- **Label**: The category label for the text (0: Politics, 1: Sport, 2: Technology, 3: Entertainment, 4: Business).

Key exploration steps include:

- **Data Overview**: Check the structure and summary statistics of the dataset.
- **Label Distribution**: Visualize the distribution of categories.
- **Text Characteristics**: Analyze text length and word frequency.

## Text Preprocessing

The preprocessing steps include:

- **Text Cleaning**: Remove mentions, URLs, and non-alphanumeric characters.
- **Feature Engineering**: Compute word counts and text lengths.

## Text Embedding Techniques

1. **Word2Vec**:

   - Train a Word2Vec model on the dataset.
   - Convert text to average Word2Vec vectors.

2. **GloVe**:

   - Load a pre-trained GloVe model.
   - Convert text to average GloVe vectors.

3. **BERT**:
   - Use a pre-trained BERT model to obtain contextualized embeddings.

## Model Training and Evaluation

For each embedding technique, a Logistic Regression model is trained and evaluated on the following metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

### Results

The performance of each embedding technique is summarized in the table below:

| Model    | Accuracy | Precision | Recall | F1 Score |
| -------- | -------- | --------- | ------ | -------- |
| Word2Vec | 0.8803   | 0.8806    | 0.8803 | 0.8792   |
| GloVe    | 0.9577   | 0.9587    | 0.9577 | 0.9580   |
| BERT     | 0.9836   | 0.9840    | 0.9836 | 0.9836   |

## Usage

1. **Install Dependencies**:

   ```bash
   pip install numpy pandas plotly seaborn matplotlib gensim scikit-learn transformers torch
   ```

2. **Prepare Dataset**:
   Ensure `data.csv` is in the same directory as the script.

3. **Run the Script**:
   Execute the script to perform data exploration, preprocessing, model training, and evaluation.

```bash
python script.py
```

## Dependencies

- **Python 3.x**
- **Libraries**:
  - `numpy`
  - `pandas`
  - `plotly`
  - `seaborn`
  - `matplotlib`
  - `gensim`
  - `scikit-learn`
  - `transformers`
  - `torch`
