import dataclasses
import logging
import os
import sys
import time

from transformers import BioGptForSequenceClassification, BioGptConfig, BioGptTokenizer

import settings
from metrics import Metric
from pipeline import Trainer
from settings.args import HfArgumentParser, ArgumentsForHiTransformer as Arguments
from dataset import MimicDataset, Collator
from settings.files import write_object_to_json_file
from settings.utils import Splits
from settings.print import print_seconds, set_logging_format

sys.path.insert(0, "/")
logger = logging.getLogger(__name__)


def parse_args():
    parser = HfArgumentParser([Arguments])
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    args._setup_devices
    set_logging_format(os.path.join(args.output_dir, "eval.log"))
    logger.info(args)
    return args


def load_data(args):
    logger.info("**************************************************")
    logger.info("*               Load the datasets                *")
    logger.info("**************************************************")
    tokenizer = BioGptTokenizer.from_pretrained(settings.BIOGPT_CHECKPOINT, use_fast=True)

    train_dataset = MimicDataset(
        tokenizer=tokenizer,
        split=Splits.train.value,
        cache_dir=settings.CACHE_DIR
    )

    idx2label = {v: k for k, v in train_dataset.label2idx.items()}
    test_dataset = MimicDataset(
        tokenizer=tokenizer,
        split=Splits.test.value,
        label2idx=train_dataset.label2idx,
        cache_dir=settings.CACHE_DIR,
    )
    return tokenizer, test_dataset, idx2label


def build_trainer(tokenizer, args, idx2label):
    if args.should_log:
        logger.info("**************************************************")
        logger.info("*               Build the trainer                *")
        logger.info("**************************************************")

    config = BioGptConfig.from_pretrained(settings.BIOGPT_CHECKPOINT, num_labels=settings.NUM_LABELS)
    model = BioGptForSequenceClassification.from_pretrained(args.model_dir, config=config)
    compute_metrics = Metric(idx2label, args.task_name)
    data_collator = Collator(tokenizer=tokenizer, max_seq_length=settings.MAX_SEQ_LENGTH)
    trainer = Trainer(model=model, args=args, data_collator=data_collator, compute_metrics=compute_metrics)

    return trainer


def main():
    args = parse_args()
    tokenizer, test_dataset, idx2label = load_data(args)
    trainer = build_trainer(tokenizer, args, idx2label)
    test_outputs = trainer.predict(test_dataset, metric_key_prefix="test")

    if args.output_predictions_filepath is not None:
        preds = Metric.get_labels_from_logitis(test_outputs["logits"], idx2label, args.task_name)
        write_object_to_json_file(preds, args.output_predictions_filepath)

    args.complete_running_time = print_seconds(time.time() - args.init_args_time)
    write_object_to_json_file(
        dict(
            args=dataclasses.asdict(args),
            test_metrics=test_outputs["metrics"],
            filepath=args.output_metrics_filepath
        )
    )


if __name__ == "__main__":
    main()
