import torch
from transformers import AutoTokenizer, BioGptForCausalLM

if __name__ == '__main__':
    model: BioGptForCausalLM = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

    model.load_state_dict(torch.load('./model_output/causal_lm_tapt_weights.bin').to('cpu'))
    # outputs = model(**inputs, labels=inputs["input_ids"])
    # loss = outputs.loss
    # logits = outputs.logits
