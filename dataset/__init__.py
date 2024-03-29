"""
Dataset implementation of MIMIC-III
https://physionet.org/content/mimiciii/1.4/
Inspired by
https://github.com/coastalcph/trldc/blob/main/dainlp/data/cls/__init__.py
"""
import json
import logging
import os

import torch
from filelock import FileLock
from torch.utils.data import Dataset

from settings.args import Arguments

logger = logging.getLogger(__name__)

"""https://github.com/coastalcph/trldc/blob/main/dainlp/data/cls/__init__.py#L9"""


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
        return batch


if __name__ == '__main__':
    pass
