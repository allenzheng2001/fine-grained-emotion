import torch
import pandas as pd
import numpy as np
import ast
from torch.utils.data import Dataset, DataLoader

mode = 'emotions'

class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        """
        Args:
            texts (list): List of text samples.
            labels (list): List of binary vectors representing labels for each text sample.
        """
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return {"text": text, "label": label}

def convert_evector(emotions_str_list):
    emotions_list = ast.literal_eval(emotions_str_list)
    emotions_vector = np.zeros(8)
    for emotion in emotions_list:
        emotions_vector[emotion] = 1
    return emotions_vector

def convert_ivector(intensities_str_list):
    intensities_list = ast.literal_eval(intensities_str_list)
    intensities_vector = np.zeros((8,3))
    for emotion, intensity in intensities_list:
        intensities_vector[emotion, intensity] = 1
    return intensities_vector.ravel() # 1d vector now

# Assuming you have train_texts and train_labels
def get_data_loader(path, batch_size = 1, label = 'emotions'):
    df = pd.read_csv(path).head(100)
    if(label == 'intensities'):
        dataset = CustomDataset(df['post'], [convert_ivector(gold_intensity_str) for gold_intensity_str in df['gold_intensities_ids']])
    else:
        dataset = CustomDataset(df['post'], [convert_evector(gold_emotion_str) for gold_emotion_str in df['gold_emotions_ids']])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def get_appraisal_data_loader(path, label = 'emotions'):
    print("Not yet supported")
    return

def get_appraisal_data_dataset(path, label = 'emotions'):
    df = pd.read_csv(path)
    input_list = np.array([ast.literal_eval(appraisal_vector_str) for appraisal_vector_str in df['appraisal_vector']])
    if(label == 'intensities'):
        dataset = list(zip(input_list, np.array([convert_ivector(gold_intensity_str) for gold_intensity_str in df['gold_intensities_ids']])))
    else:
        dataset = list(zip(input_list, np.array([convert_evector(gold_emotion_str) for gold_emotion_str in df['gold_emotions_ids']])))
    return dataset

# eventually gonna distinguish emotion/intensity...
def get_loader(mode, appraisal_flag):
    if(not appraisal_flag):
        train_loader = get_data_loader('CovidET_emotions/CovidET-ALL-train_val_test/train.csv', label = mode)
        val_loader = get_data_loader('CovidET_emotions/CovidET-ALL-train_val_test/val.csv', label = mode)
        test_loader = get_data_loader('CovidET_emotions/CovidET-ALL-train_val_test/test.csv', label = mode)
        all_loader = get_data_loader('CovidET_emotions/CovidET-ALL.csv', label = mode)
        return train_loader, val_loader, test_loader, all_loader
    else:
        train_loader_appr = get_appraisal_data_loader('CovidET_emotions/CovidET-ALL-train_val_test/train_w_appraisal.csv', label = mode)
        val_loader_appr = get_appraisal_data_loader('CovidET_emotions/CovidET-ALL-train_val_test/val_w_appraisal.csv', label = mode)
        test_loader_appr = get_appraisal_data_loader('CovidET_emotions/CovidET-ALL-train_val_test/test_w_appraisal.csv', label = mode)
        all_loader_appr = get_appraisal_data_loader('CovidET_emotions/CovidET-ALL_w_appraisal.csv', label = mode)
        return train_loader_appr, val_loader_appr, test_loader_appr, all_loader_appr

def get_datasets(mode, appraisal_flag):
        train_loader_appr = get_appraisal_data_dataset('CovidET_emotions/CovidET-ALL-train_val_test/train_w_appraisal.csv', label = mode)
        val_loader_appr = get_appraisal_data_dataset('CovidET_emotions/CovidET-ALL-train_val_test/val_w_appraisal.csv', label = mode)
        test_loader_appr = get_appraisal_data_dataset('CovidET_emotions/CovidET-ALL-train_val_test/test_w_appraisal.csv', label = mode)
        all_loader_appr = get_appraisal_data_dataset('CovidET_emotions/CovidET-ALL_w_appraisal.csv', label = mode)
        return train_loader_appr, val_loader_appr, test_loader_appr, all_loader_appr

    

