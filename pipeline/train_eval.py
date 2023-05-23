import dataclasses
import logging
import os
import time
from typing import Dict, Tuple
from datetime import datetime

import torch
from transformers import BioGptTokenizer, BioGptConfig, BioGptForSequenceClassification, BioGptModel

import settings
from dataset import MimicDataset, Collator
from metrics import Metric
from pipeline import Trainer
from pipeline.callback import EarlyStoppingCallback
from settings.args import HfArgumentParser, ArgumentsForHiTransformer as Arguments
from settings.files import write_object_to_json_file
from settings.utils import set_seed, Splits
from settings.print import set_logging_format, print_seconds

logger = logging.getLogger(__name__)


def parse_args() -> Arguments:
    parser = HfArgumentParser([Arguments])
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    device = args._setup_devices
    set_logging_format(
        os.path.join(args.output_dir, "logs", f"train_eval_{datetime.now().strftime('%m_%d_%H_%M')}.log"))
    set_seed(args.seed)
    if args.should_log:
        logger.info("**************************************************")
        logger.info("*             Parse the arguments                *")
        logger.info("**************************************************")
        logger.info(f"Device: {device}")
        logger.info(args)
    return args


def load_data(args: Arguments) -> Tuple:
    if args.should_log:
        logger.info("**************************************************")
        logger.info("*               Load the datasets                *")
        logger.info("**************************************************")

    tokenizer = BioGptTokenizer.from_pretrained(settings.BIOGPT_CHECKPOINT, use_fast=True)
    data_collator = Collator(tokenizer=tokenizer, max_seq_length=settings.MAX_SEQ_LENGTH)

    train_dataset: MimicDataset = MimicDataset(
        tokenizer=tokenizer,
        split=Splits.train.value,
        cache_dir=args.cache_dir or settings.CACHE_DIR
    )

    dev_dataset: MimicDataset = MimicDataset(
        tokenizer=tokenizer,
        split=Splits.dev.value,
        label2idx=train_dataset.label2idx,
        cache_dir=args.cache_dir or settings.CACHE_DIR
    )

    test_dataset: MimicDataset = MimicDataset(
        tokenizer=tokenizer,
        split=Splits.test.value,
        label2idx=train_dataset.label2idx,
        cache_dir=args.cache_dir or settings.CACHE_DIR,
    )

    idx2label: Dict = dict((v, k) for k, v in train_dataset.label2idx.items())

    return tokenizer, train_dataset, dev_dataset, test_dataset, idx2label, data_collator


def build_trainer(tokenizer, train_dataset, dev_dataset, args, idx2label, data_collator) -> Tuple[Trainer, BioGptModel]:
    if args.should_log:
        logger.info("**************************************************")
        logger.info("*               Build the trainer                *")
        logger.info("**************************************************")

    config: BioGptConfig = BioGptConfig.from_pretrained(settings.BIOGPT_CHECKPOINT, num_labels=settings.NUM_LABELS)
    model: BioGptModel = BioGptForSequenceClassification.from_pretrained(settings.BIOGPT_CHECKPOINT, config=config)
    compute_metrics: Metric = Metric(idx2label, args.task_name)
    trainer: Trainer = Trainer(model=model, args=args, data_collator=data_collator, train_dataset=train_dataset,
                               eval_dataset=dev_dataset, tokenizer=tokenizer, compute_metrics=compute_metrics,
                               callbacks=[EarlyStoppingCallback])

    return trainer, model


def build_evaluator(model, args, idx2label, data_collator) -> Tuple[Trainer, BioGptModel]:
    if args.should_log:
        logger.info("**************************************************")
        logger.info("*               Build the evaluator              *")
        logger.info("**************************************************")

    compute_metrics: Metric = Metric(idx2label, args.task_name)
    evaluator: Trainer = Trainer(model=model,
                                 args=args,
                                 data_collator=data_collator,
                                 compute_metrics=compute_metrics)

    return evaluator, model


def train(tokenizer, train_dataset, dev_dataset, args, idx2label, data_collator) -> BioGptModel:
    if args.should_log:
        logger.info("**************************************************")
        logger.info("*               Starting training                *")
        logger.info("**************************************************")

    trainer, model = build_trainer(tokenizer, train_dataset, dev_dataset, args, idx2label, data_collator)
    train_metrics = trainer.train()
    dev_metrics = trainer.predict(dev_dataset, metric_key_prefix="dev")["metrics"]

    args.complete_running_time = print_seconds(time.time() - args.init_args_time)

    write_object_to_json_file(
        data=dict(
            args=dataclasses.asdict(args),
            training_state=dataclasses.asdict(trainer.state),
            train_metrics=train_metrics,
            dev_metrics=dev_metrics
        ),
        filepath=args.output_metrics_train_filepath,
        sort_keys=True
    )

    return model


def evaluate(model, args, idx2label, data_collator) -> BioGptModel:
    if args.should_log:
        logger.info("**************************************************")
        logger.info("*               Starting evaluation              *")
        logger.info("**************************************************")

    evaluator, model = build_evaluator(
        model=model,
        args=args,
        idx2label=idx2label,
        data_collator=data_collator
    )
    test_outputs = evaluator.predict(test_dataset, metric_key_prefix="test")

    if args.output_predictions_filepath is not None:
        preds = Metric.get_labels_from_logitis(test_outputs["logits"], idx2label, args.task_name)
        write_object_to_json_file(preds, args.output_predictions_filepath)

    args.complete_running_time = print_seconds(time.time() - args.init_args_time)
    write_object_to_json_file(
        dict(
            args=dataclasses.asdict(args),
            test_metrics=test_outputs["metrics"],
            model_state_dict=model.state_dict()
        ),
        filepath=args.output_metrics_train_filepath,
        sort_keys=True
    )

    return model


if __name__ == '__main__':
    args: Arguments = parse_args()
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

    model = evaluate(
        model=model,
        **kwargs
    )

    save_dir = os.path.join(args.output_dir, f"runs_{datetime.now().strftime('%m_%d')}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    torch.save(
        model.state_dict(),
        os.path.join(save_dir,
                     f"pytorch_model_length_{settings.MAX_SEQ_LENGTH}_{datetime.now().strftime('%H_%M_%S')}.bin")
    )

    if args.should_log:
        logger.info("**************************************************")
        logger.info("*               Finished execution               *")
        logger.info("**************************************************")
