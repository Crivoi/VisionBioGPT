from enum import Enum

import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader

SEED = 42


class ModelCheckpoint(Enum):
    BioGPT = "microsoft/biogpt"
    BioBert = "dmis-lab/biobert-v1.1"


def train_test_split(dataset, train_ratio=.8):
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    split = int(train_ratio * num_samples)
    train_indices, test_indices = indices[:split], indices[split:]

    train_loader = DataLoader(dataset, sampler=SubsetRandomSampler(train_indices), batch_size=dataset.batch_size)
    test_loader = DataLoader(dataset, sampler=SubsetRandomSampler(test_indices), batch_size=dataset.batch_size)
    return train_loader, test_loader
