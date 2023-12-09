import numpy as np
import pandas as pd
import torch
from tqdm import tqdm as tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_lr(lr, test_set):
    all_labels = np.zeros(1)
    all_predictions = np.zeros(1)
    for x_i, y_i in tqdm(test_set):
        y_pred = lr.predict(x_i)
        
        all_labels = np.concatenate((all_labels, y_i), axis = 0)
        all_predictions = np.concatenate((all_predictions, y_pred), axis = 0)

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    # Print evaluation metrics
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

def eval_bart(model, tokenizer, test_set):
    all_l = [0]
    all_p = [0]
    with torch.no_grad():
        for ex in test_set:
            input = ex["text"]
            input_tokens = (tokenizer(input, return_tensors='pt', truncation=True, padding=True)).to(device)
            labels = (ex["label"]).to(device)

            # Forward pass
            outputs = model(**input_tokens)
            logits = outputs.logits

            # Convert logits to binary predictions
            predictions = (torch.sigmoid(logits) > 0.5).float()

            # Collect labels and predictions
            all_l += list(labels.cpu().numpy())
            all_p += list(predictions.cpu().numpy())

    print(all_l)
    print(all_p)

    # Convert lists to NumPy arrays
    all_labels = np.array(all_l)
    all_predictions = np.array(all_p)

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions,)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    # Print evaluation metrics
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")