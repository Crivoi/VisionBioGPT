import dataclasses
import logging
import os
import sys
import time
import dainlp

from transformers import AutoTokenizer, AutoConfig, BioGptForSequenceClassification, BioGptConfig, BioGptTokenizer

import settings
from dainlp.metrics.cls import Metric
from dainlp.training import Trainer
from dainlp.training.callback import EarlyStoppingCallback
from dainlp.utils.args import HfArgumentParser, ArgumentsForHiTransformer as Arguments
from dainlp.utils.print import print_seconds
from dataset import MimicDataset, Collator
from settings.files import write_object_to_json_file
from settings.utils import Splits

sys.path.insert(0, "../")

logger = logging.getLogger(__name__)


def parse_args():
    parser = HfArgumentParser([Arguments])
    args = parser.parse_args_into_dataclasses()[0]
    args._setup_devices
    dainlp.utils.print.set_logging_format(os.path.join(args.output_dir, "training.log"))
    dainlp.utils.set_seed(args.seed)
    if args.should_log:
        # logger.info(f"DaiNLP {dainlp.__version__}")
        logger.info("**************************************************")
        logger.info("*             Parse the arguments                *")
        logger.info("**************************************************")
        logger.info(args)
    return args


def load_data(args):
    if args.should_log:
        logger.info("**************************************************")
        logger.info("*               Load the datasets                *")
        logger.info("**************************************************")
    tokenizer = BioGptTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)

    train_dataset = MimicDataset(
        tokenizer=tokenizer,
        split=Splits.train.value,
        cache_dir=args.cache_dir
    )
    dev_dataset = MimicDataset(
        tokenizer=tokenizer,
        split=Splits.dev.value,
        label2idx=train_dataset.label2idx,
        cache_dir=args.cache_dir
    )

    idx2label = {v: k for k, v in train_dataset.label2idx.items()}
    return tokenizer, train_dataset, dev_dataset, idx2label


def build_trainer(tokenizer, train_dataset, dev_dataset, args, idx2label):
    if args.should_log:
        logger.info("**************************************************")
        logger.info("*               Build the trainer                *")
        logger.info("**************************************************")

    config = BioGptConfig.from_pretrained(settings.BIOGPT_CHECKPOINT, num_labels=settings.NUM_LABELS)
    model = BioGptForSequenceClassification.from_pretrained(settings.BIOGPT_CHECKPOINT, config=config)
    compute_metrics = Metric(idx2label, args.task_name)
    data_collator = Collator(tokenizer=tokenizer, max_seq_length=settings.MAX_SEQ_LENGTH)
    trainer = Trainer(model=model, args=args, data_collator=data_collator, train_dataset=train_dataset,
                      eval_dataset=dev_dataset, tokenizer=tokenizer, compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback])

    return trainer


def main():
    args = parse_args()
    tokenizer, train_dataset, dev_dataset, idx2label = load_data(args)
    trainer = build_trainer(tokenizer, train_dataset, dev_dataset, args, idx2label)
    train_metrics = trainer.train()
    dev_metrics = trainer.predict(dev_dataset, metric_key_prefix="dev")["metrics"]

    if args.should_log:
        args.complete_running_time = print_seconds(time.time() - args.init_args_time)
        write_object_to_json_file(
            data=dict(
                args=dataclasses.asdict(args),
                training_state=dataclasses.asdict(trainer.state),
                train_metrics=train_metrics,
                dev_metrics=dev_metrics
            ),
            filepath=args.output_metrics_filepath,
            sort_keys=True
        )


if __name__ == "__main__":
    main()
