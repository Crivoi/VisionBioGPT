"""
Dataset implementation of MIMIC-CXR
https://physionet.org/content/mimic-cxr/2.0.0/
Inspired by
https://github.com/coastalcph/trldc/blob/main/dainlp/data/cls/__init__.py
"""
import json
import logging
import os
import random

import torch
from PIL import Image
from filelock import FileLock
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BioGptTokenizer
from transformers.image_processing_utils import BaseImageProcessor

from settings import CXR_DATA_DIR
from settings.args import Arguments
from settings.utils import MimicCXRLabels, ViewPositionTokens

logger = logging.getLogger(__name__)


class MimicCXRDataset(Dataset):
    task_name: str = 'multilabel'
    do_lower_case: bool = False
    max_seq_length: int
    data_dir: str
    cache_dir: str
    processor: BaseImageProcessor
    tokenizer: BioGptTokenizer

    def __init__(self, args: Arguments = None, tokenizer=None, split=None, label2idx=None):
        assert (label2idx is not None) or (split == "train")

        self.data_dir = CXR_DATA_DIR
        self.cache_dir = os.path.join(self.data_dir, args.cache_dir)
        self.max_seq_length = args.max_seq_length

        self.tokenizer = tokenizer

        filepath = os.path.join(self.data_dir, f"{split}.json")
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            with FileLock(os.path.join(self.cache_dir, f"{split}.lock")):
                if os.path.exists(os.path.join(self.cache_dir, f"{split}.features")):
                    self.examples = torch.load(os.path.join(self.cache_dir, f"{split}.examples"))
                    self.label2idx = torch.load(os.path.join(self.cache_dir, f"{split}.label2idx"))
                    self.features = torch.load(os.path.join(self.cache_dir, f"{split}.features"))
                    logger.info(
                        f"Loading {len(self.features)} examples from cached directory {self.cache_dir}, split {split}"
                    )
                else:
                    self.load_from_filepath(filepath)
                    torch.save(self.examples, os.path.join(self.cache_dir, f"{split}.examples"))
                    torch.save(self.label2idx, os.path.join(self.cache_dir, f"{split}.label2idx"))
                    torch.save(self.features, os.path.join(self.cache_dir, f"{split}.features"))
        else:
            self.load_from_filepath(filepath)

    def load_from_filepath(self, filepath):
        self.examples = [json.loads(l.strip()) for l in open(filepath).readlines()][0]
        self.label2idx = {code: i for i, code in enumerate([label.value for label in MimicCXRLabels] +
                                                           [f'-{label.value}' for label in MimicCXRLabels])}
        self.features = self.convert_examples_to_features(self.examples)
        logger.info(f"Loading {len(self.features)} examples from file {filepath}")

    def convert_examples_to_features(self, examples, text_field="text"):
        features = []
        for example in tqdm(examples):
            assert len(example["path"]) == len(example["view_position"])
            assert 2 * len(example["labels"]) == len(example["labels_encoded"])

            index = random.randint(0, len(example["path"]) - 1)
            view_position = ViewPositionTokens[example["view_position"][index]]
            text = f"<{view_position}> {example[text_field]}"
            text_outputs = self.tokenizer(
                text=text,
                padding='max_length',
                truncation=True,
                max_length=self.max_seq_length
            )

            feature = dict(
                input_ids=text_outputs["input_ids"],
                attention_mask=text_outputs["attention_mask"],
                path=example["path"][index],
                labels=example["labels_encoded"],
            )

            features.append(feature)

        return features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]


class CollatorForCXR:
    def __init__(self, tokenizer, max_seq_length, processor=None, task_name="multilabel", args=None):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.task_name = task_name
        self.processor = processor
        self.data_dir = CXR_DATA_DIR
        self.args = args

    def __call__(self, features):
        max_seq_length = max([len(f["input_ids"]) for f in features])
        max_seq_length = min(max_seq_length, self.max_seq_length)

        batch = self.tokenizer.pad(features, padding=True, max_length=max_seq_length)

        if self.args and self.args.do_pixel_values:
            for k in ['input_ids', 'attention_mask']:
                batch[f'decoder_{k}'] = torch.tensor(batch[k], dtype=torch.int64)
                batch.pop(k)

            batch['pixel_values'] = list(
                map(lambda path: self.extract_img_features(os.path.join(self.data_dir, path)), batch['path'])
            )
            batch['pixel_values'] = torch.stack(batch['pixel_values']).squeeze(1)
        else:
            for k in ['input_ids', 'attention_mask']:
                batch[k] = torch.tensor(batch[k], dtype=torch.int64)
            batch.pop('path')

        assert self.task_name in ["singlelabel", "multilabel"]
        if self.task_name == "singlelabel":
            batch["labels"] = torch.tensor([f["labels"][0] for f in features], dtype=torch.int64)
        else:
            batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.float)
        return batch

    def extract_img_features(self, path):
        with Image.open(path).convert('RGB') as img:
            img_inputs = self.processor(img, return_tensors="pt")
            return img_inputs['pixel_values']


def encode_labels(row):
    positive_encoding = []
    negative_encoding = []
    for value in row.labels:
        if value == 1.:
            positive_encoding.append(1.)
            negative_encoding.append(0.)
        elif value == 0.:
            negative_encoding.append(1.)
            positive_encoding.append(0.)
        else:
            positive_encoding.append(0.)
            negative_encoding.append(0.)
    return positive_encoding + negative_encoding


def decode_labels(labels):
    positive_encoding = labels[:len(labels) // 2]
    negative_encoding = labels[len(labels) // 2:]
    decoded = []
    for i in range(len(labels) // 2 + 1):
        label_value = 1. if positive_encoding[i] else 0. if negative_encoding[i] else None
        decoded.append(label_value)
    return decoded


if __name__ == '__main__':
    pass
