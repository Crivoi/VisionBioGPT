import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from database import MimicDatabase


def train_test_split(dataset, train_ratio=.8):
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    split = int(train_ratio * num_samples)
    train_indices, test_indices = indices[:split], indices[split:]

    train_loader = DataLoader(dataset, sampler=SubsetRandomSampler(train_indices), batch_size=dataset.batch_size)
    test_loader = DataLoader(dataset, sampler=SubsetRandomSampler(test_indices), batch_size=dataset.batch_size)
    return train_loader, test_loader


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
        rows = np.array(self.database.cursor.fetchmany(self.batch_size))
        return rows[:, 0], rows[:, 1]


mimic_dataset = MimicDataset()
train_loader, test_loader = train_test_split(mimic_dataset)

if __name__ == '__main__':
    print(mimic_dataset[0])
