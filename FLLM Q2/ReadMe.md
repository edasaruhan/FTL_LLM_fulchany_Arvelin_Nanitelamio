# Violence Classification using Fine-Tuned BERT

## Project Overview

This project involves the fine-tuning of a pre-trained BERT model to classify different types of violence in tweets. This work aligns with the United Nations Sustainable Development Goal (SDG) 16: Peace, Justice, and Strong Institutions, aiming to enhance the detection of violent content on social media platforms.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Fine-Tuning](#model-fine-tuning)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Installation

### Prerequisites

- Python 3.7 or later
- pip for package management

## Dataset

The dataset (`violence.csv`) consists of tweets labeled with different types of violence. Each tweet is categorized into one of the violence types such as physical, verbal, or emotional.

### Data Preprocessing

- Unnecessary columns (e.g., `Tweet_ID`) are removed.
- The text data (tweets) is tokenized using BERT's tokenizer.
- Violence labels are converted into numerical format.

## Exploratory Data Analysis (EDA)

Before model training, EDA is performed to understand the data distribution.

### Steps:

- Count the frequency of each violence type.
- Visualize the distribution of tweet lengths.

## Model Fine-Tuning

The BERT model (`bert-base-uncased`) is fine-tuned using the labeled tweet data.

### Fine-Tuning Process:

1. **Tokenization**: Convert tweets into tokens compatible with BERT.
2. **Label Encoding**: Map violence labels to numerical values.
3. **Model Training**: Fine-tune the pre-trained BERT model.

### Training Configuration:

- **Epochs**: 3
- **Batch Size**: 16 for training, 64 for evaluation
- **Warmup Steps**: 500
- **Weight Decay**: 0.01

## Evaluation

The model is evaluated using various metrics.

Evaluation results include performance comparisons before and after fine-tuning.

## Prediction

You can use the fine-tuned model to classify new tweets. Here's an example:

```python
new_tweet = "This is an example of physical violence."
inputs = tokenizer(new_tweet, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()

reverse_label_mapping = {v: k for k, v in label_mapping.items()}
predicted_label = reverse_label_mapping[predicted_class_id]
print(f"The predicted type of violence is: {predicted_label}")
```

## Results

### Model Performance

The fine-tuned BERT model shows strong performance in classifying violence types in tweets, as indicated by high evaluation metrics.

### Metrics:

- **Accuracy**: High accuracy in correctly identifying violence types.
- **Confusion Matrix**: Visualization of true vs. predicted labels.

## Acknowledgements

- Inspired by SDG 16: Peace, Justice, and Strong Institutions.
- Uses Hugging Face's Transformers library.
