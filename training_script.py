import torch
from sentence_transformers import SentenceTransformer, util
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = r"C:\Users\bhava\Desktop\Iportant\chatbot-project\chatbotdataset.json"

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)
if 'input' not in df.columns or 'output' not in df.columns:
    raise ValueError("The JSON must contain 'input' and 'output' keys.")

sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(device)
df['input_embedding'] = df['input'].apply(lambda x: sbert_model.encode(x, convert_to_tensor=True, device=device))

df['label'] = range(len(df))
role_prompt = "You are a virtual assistant for the CSE department. Answer questions related to courses, faculty, labs, and student support."
df['input'] = role_prompt + " " + df['input']

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(df)).to(device)

train_encodings = tokenizer(list(train_df['input']), truncation=True, padding='max_length', max_length=128)
val_encodings = tokenizer(list(val_df['input']), truncation=True, padding='max_length', max_length=128)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_encodings, train_df['label'].tolist())
val_dataset = CustomDataset(val_encodings, val_df['label'].tolist())

training_args = TrainingArguments(
    output_dir=r'C:\Users\bhava\Desktop\Iportant\chatbot-project\results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=r'C:\Users\bhava\Desktop\Iportant\chatbot-project\logs',
    logging_steps=10,
)

trainer = Trainer(
    model=distilbert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

model_save_path = r"C:\Users\bhava\Desktop\Iportant\chatbot-project\fine_tuned_distilbert_model.pt"
torch.save(distilbert_model.state_dict(), model_save_path)

print(f"✅ Model saved successfully at: {model_save_path}")
