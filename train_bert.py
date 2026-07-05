import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)

from torch.utils.data import Dataset


df = pd.read_csv("dataset/twitter_airline_sentiment.csv")
print(df.head())


df = df[['text', 'airline_sentiment']]
df.columns = ['text', 'label']


df = df.dropna()
df = df.drop_duplicates()
df['text'] = df['text'].str.lower()


label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

df['label'] = df['label'].map(label_map)

print(df['label'].value_counts())


train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'],
    df['label'],
    test_size=0.2,
    random_state=42
)


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(
    list(train_texts),
    truncation=True,
    padding=True,
    max_length=128
)

test_encodings = tokenizer(
    list(test_texts),
    truncation=True,
    padding=True,
    max_length=128
)


class TweetDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = TweetDataset(train_encodings, train_labels.tolist())
test_dataset = TweetDataset(test_encodings, test_labels.tolist())


model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)

print("Model loaded successfully")


def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# FIXED TRAINING ARGUMENTS (COMPATIBLE VERSION)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)


trainer.train()


results = trainer.evaluate()
print(results)


model.save_pretrained("saved_bert_model")
tokenizer.save_pretrained("saved_bert_model")


preds_output = trainer.predict(test_dataset)
y_pred = np.argmax(preds_output.predictions, axis=1)
y_true = preds_output.label_ids

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


df['label'].value_counts().plot(kind='bar')
plt.title("Class Distribution")
plt.show()