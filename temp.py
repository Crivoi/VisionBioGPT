import logging
import os
from datetime import datetime

import torch

import settings
from pipeline.train_eval import parse_args, load_data, train, evaluate
from settings.args import ArgumentsForHiTransformer as Arguments

logger = logging.getLogger(__name__)

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
            f"model_seq_{args.max_seq_length}_labels_{settings.NUM_LABELS}_{datetime.now().strftime('%H%M')}.bin"
        )
    )

    if args.should_log:
        logger.info("**************************************************")
        logger.info("*               Finished execution               *")
        logger.info("**************************************************")
