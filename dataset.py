from typing import Dict

import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from database import MimicDatabase


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
        return rows[:, 0], rows[:, 1]


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
train_loader: DataLoader
test_loader: DataLoader
train_loader, test_loader = train_test_split(mimic_dataset)

if __name__ == '__main__':
    print(mimic_dataset[0])
