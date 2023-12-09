import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import BartForSequenceClassification, BartTokenizer, AdamW
from tqdm import tqdm as tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replace 'facebook/bart-large-cnn' with the correct model name or path
model_name = 'facebook/bart-large-cnn'
# this is a per-dimension binary classification.
# things tried ... multiclass?

def train_bart(train_loader, label_type):
    print("Using device:", device)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForSequenceClassification.from_pretrained(model_name, num_labels= 8 if label_type == 'emotions' else 24)
    model.to(device)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.BCEWithLogitsLoss()
    num_epochs = 1

    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        for ex in train_loader:
            input = ex["text"]
            input_tokens = (tokenizer(input, return_tensors='pt', truncation=True, padding=True)).to(device)
            label = (ex["label"]).to(device)

            optimizer.zero_grad()
            output = model(**input_tokens)
            loss = criterion(output.logits, label.float())
            loss.backward()
            
            total_loss += loss.item()

            optimizer.step()

        print(f"LOSS THIS EPOCH: {total_loss}")

    return model, tokenizer

# adjust threshold based on val?
        

