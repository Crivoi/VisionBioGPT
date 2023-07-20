import os

from transformers import ViTForImageClassification, VisionEncoderDecoderModel, ViTImageProcessor, ViTModel, \
    BioGptTokenizer, ViTConfig, BioGptConfig, VisionEncoderDecoderConfig, BioGptForCausalLM, GPT2Config, GPT2Tokenizer, \
    GPT2Model, OPTModel, XGLMModel, BioGptModel

import requests
from PIL import Image
import torch

import settings
from dataset import MimicCXRDataset
from model.modeling_biogpt import BioGptForCausalLMWithCrossAttention, BioGptConfigWithCrossAttention, \
    BioGptForSequenceClassificationWithCrossAttention
from settings.utils import Splits
from settings.args import ArgumentsForHiTransformer as Arguments

if __name__ == '__main__':
    tokenizer = BioGptTokenizer.from_pretrained(settings.BIOGPT_CHECKPOINT, use_fast=True)
    args = Arguments(
        task_name='multilabel',
        cache_dir='sample',
        seed=52,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        max_seq_length=5,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        metric_for_best_model='micro_f1',
    )
    train_dataset: MimicCXRDataset = MimicCXRDataset(
        args=args,
        tokenizer=tokenizer,
        split=Splits.train.value,
    )

    processor = ViTImageProcessor.from_pretrained(settings.VIT_CHECKPOINT)
    vit_config = ViTConfig.from_pretrained(settings.VIT_CHECKPOINT)
    vit_model = ViTModel.from_pretrained(settings.VIT_CHECKPOINT, config=vit_config)

    tokenizer = BioGptTokenizer.from_pretrained(settings.BIOGPT_CHECKPOINT)
    biogpt_config = BioGptConfig.from_pretrained(
        settings.BIOGPT_CHECKPOINT,
        num_labels=28,
        label2id=train_dataset.label2idx,
        id2label={v: k for k, v in train_dataset.label2idx.items()},
        cross_attention_hidden_size=384 * 2,
        add_cross_attention=True
    )
    biogpt_config = BioGptConfigWithCrossAttention(
        cross_attention_reduce_factor=1,
        **biogpt_config.__dict__,
    )
    biogpt_model = BioGptForSequenceClassificationWithCrossAttention(biogpt_config)

    ved_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(vit_config, biogpt_config)
    ved_config.decoder_start_token_id = tokenizer.bos_token_id
    ved_config.pad_token_id = tokenizer.pad_token_id
    ved_model = VisionEncoderDecoderModel(encoder=vit_model, decoder=biogpt_model, config=ved_config)

    img_path = os.path.join('../data', 'mimic-cxr', 'files_resized', 'p10', 'p10000032', 's50414267',
                            '02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg')
    img = Image.open(img_path).convert('RGB')

    img_inputs = processor(img, return_tensors="pt")
    text = "hello world"
    text_inputs = tokenizer(text, return_tensors='pt')
    outputs = ved_model(pixel_values=img_inputs['pixel_values'], decoder_input_ids=text_inputs['input_ids'])

    pass
