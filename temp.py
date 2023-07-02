import os

from transformers import ViTForImageClassification, VisionEncoderDecoderModel, ViTImageProcessor, ViTModel, \
    BioGptTokenizer, ViTConfig, BioGptConfig, VisionEncoderDecoderConfig, BioGptForCausalLM, GPT2Config, GPT2Tokenizer, \
    GPT2Model

import requests
from PIL import Image
import torch

import settings
from model.modeling_biogpt import BioGptForCausalLMWithCrossAttention

if __name__ == '__main__':
    processor = ViTImageProcessor.from_pretrained(settings.VIT_CHECKPOINT)
    vit_model = ViTModel.from_pretrained(settings.VIT_CHECKPOINT)
    tokenizer = BioGptTokenizer.from_pretrained(settings.BIOGPT_CHECKPOINT)
    # tokenizer = GPT2Tokenizer.from_pretrained(settings.GPT2_CHECKPOINT)

    vit_config = ViTConfig.from_pretrained(settings.VIT_CHECKPOINT)
    biogpt_config = BioGptConfig.from_pretrained(settings.BIOGPT_CHECKPOINT)
    biogpt_config.add_cross_attention = True

    biogpt_model = BioGptForCausalLMWithCrossAttention(biogpt_config)

    # gpt2_config = GPT2Config.from_pretrained(settings.GPT2_CHECKPOINT)
    # gpt2_config.add_cross_attention = True

    # ved_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(vit_config, gpt2_config)

    ved_config = VisionEncoderDecoderConfig(encoder=vit_model, decoder=biogpt_model)
    ved_config.decoder_start_token_id = tokenizer.bos_token_id
    ved_config.pad_token_id = tokenizer.pad_token_id
    # ved_config.pad_token_id = 0

    ved_model = VisionEncoderDecoderModel(config=ved_config)

    img_path = os.path.join('data', 'mimic-cxr', 'files_resized', 'p10', 'p10000032', 's50414267',
                            '02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg')
    img = Image.open(img_path).convert('RGB')

    inputs = processor(img, return_tensors="pt")
    text = "hello world"
    text_inputs = tokenizer(text, return_tensors='pt')
    outputs = ved_model(pixel_values=inputs['pixel_values'], labels=text_inputs['input_ids'])

    pass
