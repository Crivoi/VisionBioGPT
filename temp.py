import os

from transformers import ViTForImageClassification, VisionEncoderDecoderModel, ViTImageProcessor, ViTModel, \
    BioGptTokenizer, ViTConfig, BioGptConfig, VisionEncoderDecoderConfig, BioGptForCausalLM, GPT2Config, GPT2Tokenizer, \
    GPT2Model, OPTModel, XGLMModel

import requests
from PIL import Image
import torch

import settings
from model.modeling_biogpt import BioGptForCausalLMWithCrossAttention, BioGptConfigWithCrossAttention

if __name__ == '__main__':
    processor = ViTImageProcessor.from_pretrained(settings.VIT_CHECKPOINT)
    vit_config = ViTConfig.from_pretrained(settings.VIT_CHECKPOINT)
    vit_model = ViTModel.from_pretrained(settings.VIT_CHECKPOINT, config=vit_config)

    tokenizer = BioGptTokenizer.from_pretrained(settings.BIOGPT_CHECKPOINT)
    biogpt_config = BioGptConfig.from_pretrained(
        settings.BIOGPT_CHECKPOINT,
        cross_attention_hidden_size=384 * 2,
        add_cross_attention=True
    )
    biogpt_config = BioGptConfigWithCrossAttention(
        cross_attention_reduce_factor=4,
        **biogpt_config.__dict__,
    )
    # biogpt_config.add_cross_attention = True
    # biogpt_config.cross_attention_reduce_factor = settings.PARAMS2REDUCE_FACTOR[7]
    biogpt_model = BioGptForCausalLMWithCrossAttention(biogpt_config)

    ved_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(vit_config, biogpt_config)
    ved_config.decoder_start_token_id = tokenizer.bos_token_id
    ved_config.pad_token_id = tokenizer.pad_token_id
    ved_model = VisionEncoderDecoderModel(encoder=vit_model, decoder=biogpt_model, config=ved_config)

    # tokenizer = GPT2Tokenizer.from_pretrained(settings.GPT2_CHECKPOINT)
    # gpt2_config = GPT2Config.from_pretrained(settings.GPT2_CHECKPOINT)
    # gpt2_config.add_cross_attention = True
    # gpt2_model = GPT2Model.from_pretrained(settings.GPT2_CHECKPOINT, config=gpt2_config)
    #
    # ved_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(vit_config, gpt2_config)
    # ved_config.decoder_start_token_id = tokenizer.bos_token_id or 2
    # ved_config.pad_token_id = tokenizer.pad_token_id or 0
    # ved_model = VisionEncoderDecoderModel(encoder=vit_model, decoder=gpt2_model, config=ved_config)

    img_path = os.path.join('data', 'mimic-cxr', 'files_resized', 'p10', 'p10000032', 's50414267',
                            '02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg')
    img = Image.open(img_path).convert('RGB')

    img_inputs = processor(img, return_tensors="pt")
    text = "hello world"
    text_inputs = tokenizer(text, return_tensors='pt')
    outputs = ved_model(pixel_values=img_inputs['pixel_values'], labels=text_inputs['input_ids'])

    pass
