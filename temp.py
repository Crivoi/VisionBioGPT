import torch
from transformers import AutoTokenizer, BioGptForCausalLM

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
    model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    logits = outputs.logits
