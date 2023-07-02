import os
import json
import torch
import logging

from tqdm import tqdm
from filelock import FileLock
from torch.utils.data import Dataset
from transformers import BioGptTokenizer

import settings

from settings.args import Arguments
from settings.utils import Splits, MimicCXRLabels

logger = logging.getLogger(__name__)


class MimicDataset(Dataset):
    task_name: str = 'multilabel'
    do_lower_case: bool = False
    max_seq_length: int
    data_dir: str
    cache_dir: str

    def __init__(self, args: Arguments, tokenizer=None, split=None, label2idx=None):
        assert (label2idx is not None) or (split == "train")

        self.data_dir = args.data_dir
        self.cache_dir = args.cache_dir
        self.do_lower_case = args.do_lower_case
        self.max_seq_length = args.max_seq_length

        filepath = f"{self.data_dir}/{split}.json"

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
                    self.load_from_filepath(filepath, tokenizer, label2idx)
                    torch.save(self.examples, os.path.join(self.cache_dir, f"{split}.examples"))
                    torch.save(self.label2idx, os.path.join(self.cache_dir, f"{split}.label2idx"))
                    torch.save(self.features, os.path.join(self.cache_dir, f"{split}.features"))
        else:
            self.load_from_filepath(filepath, tokenizer, label2idx)

    def load_from_filepath(self, filepath, tokenizer, label2idx):
        self.examples = [json.loads(l.strip()) for l in open(filepath).readlines()]
        self.label2idx = label2idx if label2idx is not None else self.build_label2idx_from_examples()
        self.features = self.convert_examples_to_features(self.examples, tokenizer)
        logger.info(f"Loading {len(self.features)} examples from file {filepath}")

    def build_label2idx_from_examples(self):
        labels = set()
        for e in self.examples:
            if self.task_name == "multilabel":
                labels = labels.union(set(e["labels"]))
            elif self.task_name == "singlelabel":
                labels.add(e["label"])
        return {l: i for i, l in enumerate(sorted(labels))}

    def get_example_label(self, example):
        if self.task_name == "singlelabel":
            return [self.label2idx[example["label"]]]
        elif self.task_name == "multilabel":
            label_ids = [0] * len(self.label2idx)
            for l in example["labels"]:
                label_ids[self.label2idx[l]] = 1
            return label_ids
        raise ValueError(f"Unknown task: {self.task_name}")

    def convert_examples_to_features(self, examples, tokenizer, text_field="text"):
        features = []
        for example in examples:
            text = example[text_field]
            if self.do_lower_case:
                text = text.lower()
            outputs = tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_seq_length
            )
            feature = dict(
                input_ids=outputs["input_ids"],
                labels=self.get_example_label(example),
                attention_mask=outputs["attention_mask"]
            )
            features.append(feature)

        # if len(examples) > 0:
        #     logger.info(features[0].get('input_ids').size)

        return features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]


class Collator:
    def __init__(self, tokenizer, max_seq_length, task_name="multilabel"):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.task_name = task_name

    def __call__(self, features):
        max_seq_length = max([len(f["input_ids"]) for f in features])
        max_seq_length = min(max_seq_length, self.max_seq_length)
        batch = self.tokenizer.pad(features, padding=True, max_length=max_seq_length)
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}

        assert self.task_name in ["singlelabel", "multilabel"]
        if self.task_name == "singlelabel":
            batch["labels"] = torch.tensor([f["labels"][0] for f in features], dtype=torch.int64)
        else:
            batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.float)
        return batch


class MimicDatasetForLM(MimicDataset):
    def convert_examples_to_features(self, examples, tokenizer, text_field="text"):
        features = []
        for example in examples:
            text = example[text_field]
            if self.do_lower_case:
                text = text.lower()
            outputs = tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_seq_length
            )
            feature = dict(
                **outputs,
                labels=outputs['input_ids']
            )
            features.append(feature)

        if len(examples) > 0:
            logger.info(examples[0])
            logger.info(features[0])

        return features


class CollatorForLM(Collator):
    def __call__(self, features):
        max_seq_length = max([len(f["input_ids"]) for f in features])
        max_seq_length = min(max_seq_length, self.max_seq_length)
        batch = self.tokenizer.pad(features, padding=True, max_length=max_seq_length)
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        # assert self.task_name in ["singlelabel", "multilabel"]
        # if self.task_name == "singlelabel":
        #     batch["labels"] = torch.tensor([f["labels"][0] for f in features], dtype=torch.int64)
        # else:
        #     batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.int64)
        return batch


class MimicCXRDataset(Dataset):
    task_name: str = 'multilabel'
    do_lower_case: bool = False
    max_seq_length: int
    data_dir: str
    cache_dir: str

    def __init__(self, args: Arguments = None, tokenizer=None, split=None, label2idx=None):
        assert (label2idx is not None) or (split == "train")

        self.files_dir = os.path.join(os.path.abspath('D:'), 'MIMIC-CXR-JPG', 'files_resized')
        self.data_dir = '../data/mimic-cxr'
        self.cache_dir = os.path.join(self.data_dir, args.cache_dir)
        self.max_seq_length = args.max_seq_length or 1024

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
                    self.load_from_filepath(filepath, tokenizer)
                    torch.save(self.examples, os.path.join(self.cache_dir, f"{split}.examples"))
                    torch.save(self.label2idx, os.path.join(self.cache_dir, f"{split}.label2idx"))
                    torch.save(self.features, os.path.join(self.cache_dir, f"{split}.features"))
        else:
            self.load_from_filepath(filepath, tokenizer)

    def load_from_filepath(self, filepath, tokenizer):
        self.examples = [json.loads(l.strip()) for l in open(filepath).readlines()][0]
        self.label2idx = {code: i for i, code in enumerate([label.value for label in MimicCXRLabels] +
                                                           [f'-{label.value}' for label in MimicCXRLabels])}

        self.features = self.convert_examples_to_features(self.examples, tokenizer)
        logger.info(f"Loading {len(self.features)} examples from file {filepath}")

    def convert_examples_to_features(self, examples, tokenizer, text_field="text"):
        features = []
        for example in tqdm(examples):
            text = example[text_field]
            assert len(example["path"]) == len(example["view_position"])
            assert 2 * len(example["labels"]) == len(example["labels_encoded"])

            outputs = tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_seq_length
            )

            feature = dict(
                input_ids=outputs["input_ids"],
                labels=example["labels_encoded"],
                path=example["path"],
                view_position=example["view_position"],
                attention_mask=outputs["attention_mask"]
            )

            features.append(feature)

        return features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]


class CollatorForCXR:
    def __init__(self, tokenizer, max_seq_length, task_name="multilabel"):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.task_name = task_name

    def __call__(self, features):
        max_seq_length = max([len(f["input_ids"]) for f in features])
        max_seq_length = min(max_seq_length, self.max_seq_length)

        batch = self.tokenizer.pad(features, padding=True, max_length=max_seq_length)
        batch = {k: batch[k] for k in ['input_ids', 'labels', 'attention_mask']}
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}

        assert self.task_name in ["singlelabel", "multilabel"]
        if self.task_name == "singlelabel":
            batch["labels"] = torch.tensor([f["labels"][0] for f in features], dtype=torch.int64)
        else:
            batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.float)
        return batch


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
    tokenizer = BioGptTokenizer.from_pretrained(settings.BIOGPT_CHECKPOINT, use_fast=True)
    train_dataset = MimicCXRDataset(tokenizer=tokenizer, split=Splits.train.value)
    idx2label = {v: k for k, v in train_dataset.label2idx.items()}
