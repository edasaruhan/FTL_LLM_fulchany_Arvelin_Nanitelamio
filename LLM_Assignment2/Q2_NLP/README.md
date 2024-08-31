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
7. [Conclusion](#conclusion)
8. [Usage](#usage)
9. [Dependencies](#dependencies)
10. [Dataset Source](#dataset-source)

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


## Conclusion

In this text classification project, we aimed to classify text samples into five distinct categories using various text embedding techniques. Through a detailed exploration, preprocessing, and evaluation process, we compared the performance of three prominent embedding methods: Word2Vec, GloVe, and BERT.

**Key Findings:**

- **Word2Vec**: Achieved an accuracy of 0.8803 with balanced performance across precision, recall, and F1 score. While it provided a solid baseline, its performance was outshone by more advanced techniques.

- **GloVe**: Demonstrated significant improvement over Word2Vec with an accuracy of 0.9577. Its ability to leverage pre-trained embeddings resulted in enhanced precision, recall, and F1 scores, making it a robust choice for text classification tasks.

- **BERT**: Outperformed both Word2Vec and GloVe, achieving the highest accuracy of 0.9836. BERT’s contextual embeddings proved highly effective in understanding and classifying text with nuanced meanings and complex contexts.

**Implications:**

The results underscore BERT’s superiority in handling text classification tasks due to its deep contextual understanding of language. For applications requiring high accuracy and intricate language comprehension, BERT is the preferred choice. GloVe offers a balance between performance and computational efficiency, while Word2Vec remains useful for simpler tasks or scenarios with limited resources.

**Future Work:**

- **Model Fine-Tuning**: Further tuning and exploring additional hyperparameters could potentially enhance the performance of the models.
- **Additional Techniques**: Investigating other advanced models or techniques, such as Transformer-based architectures beyond BERT, might provide further improvements.
- **Application-Specific Adjustments**: Tailoring models to specific domains or types of text could yield more specialized and accurate results.

Overall, this project demonstrates the effectiveness of modern embedding techniques in text classification and highlights the importance of choosing the right model based on specific needs and available resources.

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
 
## Dataset Source

The dataset used for this project can be accessed [here](https://www.kaggle.com/datasets/sunilthite/text-document-classification-dataset/data).
