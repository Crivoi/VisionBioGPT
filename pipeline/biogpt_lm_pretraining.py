import dataclasses
import logging
import os
import time
from typing import Dict, Tuple
from datetime import datetime

import torch
import wandb
from transformers import BioGptTokenizer, BioGptConfig, BioGptModel, BioGptForCausalLM

import settings
from dataset import MimicDataset, MimicDatasetForLM, CollatorForLM
from metrics import Metric
from pipeline import Trainer
from pipeline.callback import EarlyStoppingCallback
from settings.args import HfArgumentParser, ArgumentsForHiTransformer as Arguments
from settings.files import write_object_to_json_file
from settings.utils import set_seed, Splits
from settings.print import set_logging_format, print_seconds, log_metrics

logger = logging.getLogger(__name__)


def parse_args() -> Arguments:
    parser = HfArgumentParser([Arguments])
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    device = args._setup_devices
    set_logging_format(
        os.path.join(args.output_dir, "lm_training_logs", f"lm_train_{datetime.now().strftime('%H%M')}.log"))
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
    data_collator = CollatorForLM(tokenizer=tokenizer, max_seq_length=args.max_seq_length)

    train_dataset: MimicDataset = MimicDatasetForLM(
        args=args,
        tokenizer=tokenizer,
        split=Splits.train.value,
    )

    dev_dataset: MimicDataset = MimicDatasetForLM(
        args=args,
        tokenizer=tokenizer,
        split=Splits.dev.value,
        label2idx=train_dataset.label2idx,
    )

    test_dataset: MimicDataset = MimicDatasetForLM(
        args=args,
        tokenizer=tokenizer,
        split=Splits.test.value,
        label2idx=train_dataset.label2idx,
    )

    idx2label: Dict = dict((v, k) for k, v in train_dataset.label2idx.items())

    return tokenizer, train_dataset, dev_dataset, test_dataset, idx2label, data_collator


def build_trainer(tokenizer, train_dataset, dev_dataset, args, idx2label, data_collator) -> Tuple[Trainer, BioGptModel]:
    if args.should_log:
        logger.info("**************************************************")
        logger.info("*               Build the trainer                *")
        logger.info("**************************************************")

    config: BioGptConfig = BioGptConfig.from_pretrained(settings.BIOGPT_CHECKPOINT, num_labels=settings.NUM_LABELS)
    model: BioGptModel = BioGptForCausalLM.from_pretrained(settings.BIOGPT_CHECKPOINT, config=config)
    # compute_metrics: Metric = Metric(idx2label, args.task_name)
    trainer: Trainer = Trainer(model=model, args=args, data_collator=data_collator, train_dataset=train_dataset,
                               eval_dataset=dev_dataset, tokenizer=tokenizer, callbacks=[EarlyStoppingCallback])

    return trainer, model


def train(tokenizer, train_dataset, dev_dataset, args, idx2label, data_collator) -> BioGptModel:
    if args.should_log:
        logger.info("**************************************************")
        logger.info("*               Starting training                *")
        logger.info("**************************************************")

    trainer, model = build_trainer(tokenizer, train_dataset, dev_dataset, args, idx2label, data_collator)

    ppl_before_training = trainer.compute_perplexity(dev_dataset)

    logger.info(f"Perplexity on test set before training: {ppl_before_training['perplexity']}")

    train_metrics = trainer.train()
    # dev_metrics = trainer.predict(dev_dataset, metric_key_prefix="dev")["metrics"]

    ppl_after_training = trainer.compute_perplexity(dev_dataset)

    logger.info(f"Perplexity on test set after training: {ppl_after_training['perplexity']}")

    args.complete_running_time = print_seconds(time.time() - args.init_args_time)

    log_metrics(split=Splits.train.value, metrics=train_metrics)
    # log_metrics(split=Splits.dev.value, metrics=dev_metrics)

    write_object_to_json_file(
        data=dict(
            args=dataclasses.asdict(args),
            training_state=dataclasses.asdict(trainer.state),
            train_metrics=train_metrics,
            # dev_metrics=dev_metrics,
            model_state_dict=model.state_dict()
        ),
        filepath=os.path.join(args.output_dir, "train_metrics.json"),
        sort_keys=True
    )

    return model


def main():
    wandb.login()

    args: Arguments = parse_args()

    with wandb.init(entity='andrei-crivoi1997', project="biogpt-lm-pretraining", config=args.__dict__):
        tokenizer, train_dataset, dev_dataset, test_dataset, idx2label, data_collator = load_data(args)
        kwargs = dict(
            args=args,
            idx2label=idx2label,
            data_collator=data_collator
        )

        model = train(
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            **kwargs
        )

        runs_dir = os.path.join(args.output_dir, "runs")
        if not os.path.exists(runs_dir):
            os.makedirs(runs_dir)

        model_save_dir = os.path.join(
            runs_dir,
            f"causal_lm_tapt_{args.max_seq_length}.bin"
        )

        torch.save(model.state_dict(), model_save_dir)

        if args.should_log:
            logger.info(f"Saved trained model at {model_save_dir}")
            logger.info("**************************************************")
            logger.info("*               Finished execution               *")
            logger.info("**************************************************")


if __name__ == '__main__':
    main()
