import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Load the dataset
df = pd.read_csv("violence.csv")
print(df.head())

# Exploratory Data Analysis (EDA)
print(df.columns)
print(df['type'].value_counts())

# Drop unnecessary columns
df = df.drop(columns=['Tweet_ID'], axis=1)
print(df.tail())

# Visualize the distribution of violence types
sns.countplot(y='type', data=df)
plt.title("Distribution of Violence Types")
plt.show()

# Analyze tweet length distribution
df['text_length'] = df['tweet'].apply(len)
sns.histplot(df['text_length'], kde=True)
plt.title("Distribution of Tweet Lengths")
plt.show()

# Sampling the dataset for quicker processing (optional)
df = df.sample(1000, random_state=42)
print(df['type'].value_counts())

# Tokenize the data
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenized_data = tokenizer(df['tweet'].tolist(), padding=True, truncation=True, return_tensors='pt')

# Convert labels to numerical format
label_mapping = {label: i for i, label in enumerate(df['type'].unique())}
df['label'] = df['type'].map(label_mapping)

# Add labels to the tokenized data
tokenized_data['labels'] = df['label'].tolist()

# Convert to Hugging Face dataset format
dataset = Dataset.from_dict(tokenized_data)

# Split into train and eval datasets
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_mapping))

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Train the model
trainer.train()

# Evaluate the fine-tuned model
evaluation_results = trainer.evaluate()
print(f"Evaluation Results: {evaluation_results}")

# Compare Performance: Pre-finetuning and Post-finetuning
pre_finetune_eval = trainer.evaluate(eval_dataset)
post_finetune_eval = trainer.evaluate()
print("Pre Fine-Tuning Evaluation:", pre_finetune_eval)
print("Post Fine-Tuning Evaluation:", post_finetune_eval)

# Predict violence type for a new tweet
new_tweet = "This is an example of physical violence."
inputs = tokenizer(new_tweet, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()

# Reverse mapping of labels
reverse_label_mapping = {v: k for k, v in label_mapping.items()}
predicted_label = reverse_label_mapping[predicted_class_id]
print(f"The predicted type of violence is: {predicted_label}")