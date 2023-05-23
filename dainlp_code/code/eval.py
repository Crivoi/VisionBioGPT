import dataclasses
import json
import logging
import os
import sys
import time

from transformers import AutoTokenizer, AutoConfig

from dainlp_code import dainlp
from data.cls import Dataset
from data.cls.hierarchical import Collator
from dainlp_code.dainlp.metrics.cls import Metric
from dainlp_code.dainlp.models.cls.hierarchical import DaiRobertaModel
from dainlp_code.dainlp.training import Trainer
from settings.args import HfArgumentParser, ArgumentsForHiTransformer as Arguments
from dainlp_code.dainlp.utils import print_seconds

sys.path.insert(0, "../../")
logger = logging.getLogger(__name__)


def parse_args():
    parser = HfArgumentParser([Arguments])
    args = parser.parse_args_into_dataclasses()[0]
    args._setup_devices
    dainlp.utils.print.set_logging_format(os.path.join(args.output_dir, "eval.log"))
    logger.info(f"DaiNLP {dainlp.__version__}")
    logger.info(args)
    return args


def load_data(args):
    logger.info("**************************************************")
    logger.info("*               Load the datasets                *")
    logger.info("**************************************************")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    idx2label = json.load(open(os.path.join(args.model_dir, "config.json")))["id2label"]
    idx2label = {int(k): v for k, v in idx2label.items()}
    label2idx = {v: k for k, v in idx2label.items()}
    test_dataset = Dataset(args.test_filepath, args, tokenizer, split="test", label2idx=label2idx)
    return tokenizer, test_dataset, idx2label


def build_trainer(tokenizer, args, idx2label):
    logger.info("**************************************************")
    logger.info("*               Build the trainer                *")
    logger.info("**************************************************")
    config = AutoConfig.from_pretrained(args.model_dir)
    model = DaiRobertaModel.from_pretrained(args.model_dir, config=config)
    data_collator = Collator(tokenizer, args.segment_length, args.max_num_segments,
                             args.do_use_stride, args.add_cls_each_segment, args.task_name)
    compute_metrics = Metric(idx2label, args.task_name)

    trainer = Trainer(model=model, args=args, data_collator=data_collator, compute_metrics=compute_metrics)
    return trainer


def main():
    args = parse_args()
    tokenizer, test_dataset, idx2label = load_data(args)
    trainer = build_trainer(tokenizer, args, idx2label)
    test_outputs = trainer.predict(test_dataset, metric_key_prefix="test")

    if args.output_predictions_filepath is not None:
        preds = Metric.get_labels_from_logitis(test_outputs["logits"], idx2label, args.task_name)
        dainlp.utils.files.write_object_to_json_file(preds, args.output_predictions_filepath)

    args.complete_running_time = print_seconds(time.time() - args.init_args_time)
    dainlp.utils.files.write_object_to_json_file(
        {"args": dataclasses.asdict(args), "test_metrics": test_outputs["metrics"]}, args.output_metrics_filepath)


if __name__ == "__main__":
    main()
