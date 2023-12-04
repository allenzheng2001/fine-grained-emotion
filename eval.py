import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
