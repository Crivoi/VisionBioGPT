from typing import Dict

import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from database import MimicDatabase
from preprocessing import TextProcessor
from settings.__init__ import BATCH_SIZE, DEVICE


class MimicDataset(Dataset):
    def __init__(self, batch_size=32) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.database = MimicDatabase()
        self.total_size = self.database.count_discharge_summaries

    def __len__(self):
        return self.total_size // self.batch_size

    def __getitem__(self, index):
        """
        :param index: Unused
        :return: batches of size [self.batch_size, self.max_length] representing texts and labels
        """
        self.database.query_text_and_icd9_code()
        rows = np.array(self.database.fetchmany(self.batch_size))
        texts, labels = rows[:, 0], rows[:, 1]
        tokenized = [TextProcessor.tokenize(text) for text in texts]

        input_ids = self.encode_tokens(tokenized)
        label_ids = self.encode_tokens(labels)

        return input_ids, label_ids

    def encode_tokens(self, tokens):
        return pad_sequence([TextProcessor.encode(t)
                             for t in tokens]).view((self.batch_size, -1)).to(DEVICE)


def train_test_split(dataset: MimicDataset, train_ratio: float = .8) -> Dict[str, DataLoader]:
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    split = int(train_ratio * num_samples)
    train_indices, test_indices = indices[:split], indices[split:]

    train_loader: DataLoader = DataLoader(dataset, sampler=SubsetRandomSampler(train_indices))
    test_loader: DataLoader = DataLoader(dataset, sampler=SubsetRandomSampler(test_indices))
    return dict(train=train_loader, test=test_loader)


mimic_dataset: MimicDataset = MimicDataset(batch_size=BATCH_SIZE)
mimic_loader: Dict[str, DataLoader] = train_test_split(mimic_dataset)

if __name__ == '__main__':
    train_loader = mimic_loader.get('train')
    print(len(mimic_dataset))
    print(len(mimic_loader.get('train')))
    print(len(mimic_loader.get('test')))
