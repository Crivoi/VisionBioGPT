import logging
import os
from datetime import datetime
from typing import Dict, Tuple

from transformers import BioGptTokenizer

import settings
from dataset import MimicCXRDataset, Collator
from settings.args import HfArgumentParser, ArgumentsForHiTransformer as Arguments
from settings.print import set_logging_format
from settings.utils import set_seed, Splits

logger = logging.getLogger(__name__)


def parse_args() -> Arguments:
    parser = HfArgumentParser([Arguments])
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    device = args._setup_devices
    set_logging_format(
        os.path.join(args.output_dir, "tapt_logs", f"seq_cls_tapt_{datetime.now().strftime('%H%M')}.log"))
    set_seed(args.seed)
    if args.should_log:
        logger.info("**************************************************")
        logger.info("*             Parse the arguments                *")
        logger.info("**************************************************")
        logger.info(f"Device: {device}")
        logger.info(f"Sequence Length: {args.max_seq_length}")
        logger.info(args.__dict__)
    return args


def load_data(args: Arguments) -> Tuple:
    if args.should_log:
        logger.info("**************************************************")
        logger.info("*               Load the datasets                *")
        logger.info("**************************************************")

    tokenizer = BioGptTokenizer.from_pretrained(settings.BIOGPT_CHECKPOINT, use_fast=True)
    data_collator = Collator(tokenizer=tokenizer, max_seq_length=args.max_seq_length)

    train_dataset: MimicCXRDataset = MimicCXRDataset(
        args=args,
        tokenizer=tokenizer,
        split=Splits.train.value,
    )

    test_dataset: MimicCXRDataset = MimicCXRDataset(
        args=args,
        tokenizer=tokenizer,
        split=Splits.test.value,
        label2idx=train_dataset.label2idx,
    )

    idx2label: Dict = dict((v, k) for k, v in train_dataset.label2idx.items())

    return tokenizer, train_dataset, test_dataset, idx2label, data_collator


def main():
    args: Arguments = parse_args()
    tokenizer, train_dataset, test_dataset, idx2label, data_collator = load_data(args)
    return 0


if __name__ == '__main__':
    main()
