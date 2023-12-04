import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import BartForSequenceClassification, BartTokenizer, AdamW
from tqdm import tqdm as tqdm

from load_datasets import test_loader, train_loader, val_loader, all_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Replace 'facebook/bart-large-cnn' with the correct model name or path
model_name = 'facebook/bart-large-cnn'

tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForSequenceClassification.from_pretrained(model_name, num_labels=8)
model.to(device)
# this is a per-dimension binary classification.
# things tried ... multiclass?

def train_model():
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.BCEWithLogitsLoss()
    num_epochs = 2

    for epoch in tqdm(range(num_epochs)):
        for ex in tqdm(train_loader):
            input = ex["text"]
            input_tokens = (tokenizer(input, return_tensors='pt', truncation=True, padding=True)).to(device)
            label = (ex["label"]).to(device)

            optimizer.zero_grad()
            output = model(**input_tokens)
            loss = criterion(output.logits, label.float())
            loss.backward()
            print(loss)

            optimizer.step()

train_model()
        

