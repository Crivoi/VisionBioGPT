import os

import torch
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu
from transformers import ViTImageProcessor, ViTModel, \
    BioGptTokenizer, ViTConfig, BioGptConfig, VisionEncoderDecoderConfig

import settings
from dataset.mimic_cxr import MimicCXRDataset
from model.modeling_biogpt import BioGptForCausalLMWithCrossAttention, BioGptConfigWithCrossAttention
from model.modeling_vision_encoder_decoder import BioGptViTEncoderDecoderModel
from pipeline.callback import IntervalStrategy
from settings.args import ArgumentsForHiTransformer as Arguments
from settings.utils import Splits

if __name__ == '__main__':
    tokenizer = BioGptTokenizer.from_pretrained(settings.BIOGPT_CHECKPOINT, use_fast=True)
    args = Arguments(
        task_name='multilabel',
        cache_dir='sample',
        seed=52,
        evaluation_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
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

    biogpt_config = BioGptConfig.from_pretrained(
        settings.BIOGPT_CHECKPOINT,
        cross_attention_hidden_size=384 * 2,
        add_cross_attention=True,
        use_cache=False
    )
    biogpt_config = BioGptConfigWithCrossAttention(
        cross_attention_reduce_factor=1,
        **biogpt_config.__dict__,
    )
    biogpt_model = BioGptForCausalLMWithCrossAttention(biogpt_config)

    lm_state_dict = torch.load(os.path.join(os.pardir, 'model_output', 'causal_lm_tapt_weights_cpu.bin'))
    for key in biogpt_model.state_dict():
        if key not in lm_state_dict:
            lm_state_dict[key] = biogpt_model.state_dict()[key]
    biogpt_model.load_state_dict(lm_state_dict)

    ved_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(vit_config, biogpt_config)
    ved_config.decoder_start_token_id = tokenizer.bos_token_id
    ved_config.pad_token_id = tokenizer.pad_token_id
    ved_model = BioGptViTEncoderDecoderModel(encoder=vit_model, decoder=biogpt_model, config=ved_config)

    img_path = os.path.join('')
    img = Image.open(img_path).convert('RGB')

    img_inputs = processor(img, return_tensors="pt")
    # text = "This image shows"
    # text_inputs = tokenizer(text, return_tensors='pt')
    # outputs = ved_model(pixel_values=img_inputs['pixel_values'], decoder_input_ids=text_inputs['input_ids'])
    # generate_outputs = ved_model.generate(inputs=img_inputs['pixel_values'], max_length=512)

    hypot = ""
    generate_outputs = ved_model.generate(inputs=img_inputs['pixel_values'], max_length=100)
    generated_text = tokenizer.decode(generate_outputs[0], skip_special_tokens=True)

    print(generated_text)

    print(sentence_bleu([generated_text], hypot))
