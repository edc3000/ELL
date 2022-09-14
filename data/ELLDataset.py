import torch
import numpy as np
from transformers import BertTokenizer
import os

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pretrained_model_dir = os.path.join(project_dir, "bert_base_uncased")
tokenizer = BertTokenizer.from_pretrained(pretrained_model_dir)



class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):

        self.labels = df[["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]].reset_index()
        self.texts = df[["full_text"]].reset_index()

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels.loc[idx].values[1:]).astype(float)

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return tokenizer(self.texts.loc[idx].values[1],
                        padding='max_length', max_length = 512, truncation=True,
                        return_tensors="pt")

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.texts = df[["full_text"]]

    def __len__(self):
        return len(self.texts)

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return tokenizer(self.texts.loc[idx].values[0],
                         padding='max_length', max_length=512, truncation=True,
                         return_tensors="pt")

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        return batch_texts