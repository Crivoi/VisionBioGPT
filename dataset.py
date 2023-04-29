from typing import Dict, Union, List

import numpy as np
from nltk import RegexpTokenizer
from numpy import ndarray
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from transformers import BioGptTokenizer

from database import MimicDatabase
from preprocessing import Preprocessor


class MimicDataset(Dataset):
    def __init__(self, batch_size=32) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.database = MimicDatabase()
        self.total_size = self.database.count_discharge_summaries

    def __len__(self):
        return self.total_size // self.batch_size

    def __getitem__(self, index):
        self.database.query_text_and_icd9_code()
        rows = np.array(self.database.fetchmany(self.batch_size))
        texts, labels = rows[:, 0], rows[:, 1]

        tokenized = [Preprocessor.tokenize(text) for text in texts]
        input_ids = [Preprocessor.encode(tokens) for tokens in tokenized]

        return texts, labels


def train_test_split(dataset: MimicDataset, train_ratio: float = .8) -> Dict[str, DataLoader]:
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    split = int(train_ratio * num_samples)
    train_indices, test_indices = indices[:split], indices[split:]

    train_loader: DataLoader = DataLoader(dataset, sampler=SubsetRandomSampler(train_indices),
                                          batch_size=dataset.batch_size)
    test_loader: DataLoader = DataLoader(dataset, sampler=SubsetRandomSampler(test_indices),
                                         batch_size=dataset.batch_size)
    return dict(train=train_loader, test=test_loader)


mimic_dataset: MimicDataset = MimicDataset()
mimic_loader: Dict[str, DataLoader] = train_test_split(mimic_dataset)

if __name__ == '__main__':
    train_loader = mimic_loader.get('train')
    for idx, item in enumerate(train_loader):
        print(idx, item)
        if idx > 1:
            break
