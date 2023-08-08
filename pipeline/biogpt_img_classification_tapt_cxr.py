import dataclasses
import logging
import os
import time
from datetime import datetime
from typing import Dict, Tuple

import torch
import wandb
from transformers import BioGptTokenizer, BioGptConfig, ViTImageProcessor, \
    ViTConfig, ViTModel, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

import settings
from dataset import MimicCXRDataset
from dataset.mimic_cxr import CollatorForCXR
from metrics import Metric
from model.modeling_biogpt import BioGptConfigWithCrossAttention, BioGptForSequenceClassificationWithCrossAttention
from model.modeling_vision_encoder_decoder import BioGptViTEncoderDecoderModel
from pipeline import Trainer
from pipeline.callback import EarlyStoppingCallback
from settings.args import HfArgumentParser, ArgumentsForHiTransformer as Arguments
from settings.files import write_object_to_json_file
from settings.print import set_logging_format, print_seconds, log_metrics
from settings.utils import set_seed, Splits

logger = logging.getLogger(__name__)


def parse_args() -> Arguments:
    parser = HfArgumentParser([Arguments])
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    device = args._setup_devices
    set_logging_format(
        os.path.join(args.output_dir, "tapt_cxr_jpg_logs", f"img_cls_tapt_cxr_{datetime.now().strftime('%H%M')}.log"))
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
    processor = ViTImageProcessor.from_pretrained(settings.VIT_CHECKPOINT)

    data_collator = CollatorForCXR(tokenizer=tokenizer, max_seq_length=args.max_seq_length, processor=processor)

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


def build_trainer(
        tokenizer, train_dataset, dev_dataset, args, idx2label, data_collator
) -> Tuple[Trainer, VisionEncoderDecoderModel]:
    if args.should_log:
        logger.info("**************************************************")
        logger.info("*               Build the trainer                *")
        logger.info("**************************************************")

    vit_config = ViTConfig.from_pretrained(settings.VIT_CHECKPOINT)
    vit_model = ViTModel.from_pretrained(settings.VIT_CHECKPOINT, config=vit_config)

    biogpt_config = BioGptConfig.from_pretrained(
        settings.BIOGPT_CHECKPOINT,
        num_labels=28,
        id2label=idx2label,
        label2id={v: k for k, v in idx2label.items()},
        cross_attention_hidden_size=384 * 2,
        add_cross_attention=True
    )
    biogpt_config = BioGptConfigWithCrossAttention(
        cross_attention_reduce_factor=1,
        **biogpt_config.__dict__,
    )
    biogpt_model = BioGptForSequenceClassificationWithCrossAttention(biogpt_config)

    lm_state_dict = torch.load(os.path.join(os.pardir, 'model_output', 'causal_lm_tapt_weights_cpu.bin'))
    lm_state_dict.pop('output_projection.weight')
    for key in biogpt_model.state_dict():
        if key not in lm_state_dict:
            lm_state_dict[key] = biogpt_model.state_dict()[key]
    biogpt_model.load_state_dict(lm_state_dict)

    ved_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(vit_config, biogpt_config)
    ved_config.decoder_start_token_id = tokenizer.bos_token_id
    ved_config.pad_token_id = tokenizer.pad_token_id
    ved_model = BioGptViTEncoderDecoderModel(encoder=vit_model, decoder=biogpt_model, config=ved_config)

    compute_metrics: Metric = Metric(idx2label, args.task_name)
    trainer: Trainer = Trainer(model=ved_model, args=args, data_collator=data_collator, train_dataset=train_dataset,
                               eval_dataset=dev_dataset, tokenizer=tokenizer, compute_metrics=compute_metrics,
                               callbacks=[EarlyStoppingCallback])

    return trainer, ved_model


def build_evaluator(model, args, idx2label, data_collator) -> Tuple[Trainer, VisionEncoderDecoderModel]:
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


def train(tokenizer, train_dataset, dev_dataset, args, idx2label, data_collator) -> VisionEncoderDecoderModel:
    if args.should_log:
        logger.info("**************************************************")
        logger.info("*               Starting training                *")
        logger.info("**************************************************")

    trainer, model = build_trainer(tokenizer, train_dataset, dev_dataset, args, idx2label, data_collator)
    train_metrics = trainer.train()
    dev_metrics = trainer.predict(dev_dataset, metric_key_prefix=Splits.test.value)["metrics"]

    args.complete_running_time = print_seconds(time.time() - args.init_args_time)

    log_metrics(split=Splits.train.value, metrics=train_metrics)
    log_metrics(split=Splits.test.value, metrics=dev_metrics)

    write_object_to_json_file(
        data=dict(
            args=dataclasses.asdict(args),
            training_state=dataclasses.asdict(trainer.state),
            train_metrics=train_metrics,
            dev_metrics=dev_metrics,
            model_state_dict=model.state_dict()
        ),
        filepath=os.path.join(args.output_dir, "train_metrics.json"),
        sort_keys=True
    )

    return model


def evaluate(model, test_dataset, args, idx2label, data_collator) -> VisionEncoderDecoderModel:
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

    log_metrics(split=Splits.test.value, metrics=test_outputs["metrics"])

    if args.output_predictions_filepath is not None:
        preds = Metric.get_labels_from_logits(test_outputs["logits"], idx2label, args.task_name)
        write_object_to_json_file(preds, args.output_predictions_filepath)

    args.complete_running_time = print_seconds(time.time() - args.init_args_time)
    write_object_to_json_file(
        dict(
            args=dataclasses.asdict(args),
            test_metrics=test_outputs["metrics"],
            model_state_dict=model.state_dict()
        ),
        filepath=os.path.join(args.output_dir, "test_metrics.json"),
        sort_keys=True
    )

    return model


def main():
    wandb.login()

    args: Arguments = parse_args()

    with wandb.init(entity='andrei-crivoi1997', project="biogpt-img-cls-tapt-cxr", config=args.__dict__):
        tokenizer, train_dataset, test_dataset, idx2label, data_collator = load_data(args)
        kwargs = dict(
            args=args,
            idx2label=idx2label,
            data_collator=data_collator
        )

        model = train(
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            dev_dataset=test_dataset,
            **kwargs
        )

        model = evaluate(
            model=model,
            test_dataset=test_dataset,
            **kwargs
        )

        runs_dir = os.path.join(args.output_dir, "runs")
        if not os.path.exists(runs_dir):
            os.makedirs(runs_dir)

        torch.save(
            model.state_dict(),
            os.path.join(
                runs_dir,
                f"model_img_classification_tapt_cxr_{datetime.now().strftime('%H%M')}.bin"
            )
        )

        if args.should_log:
            logger.info("**************************************************")
            logger.info("*               Finished execution               *")
            logger.info("**************************************************")


if __name__ == '__main__':
    main()
