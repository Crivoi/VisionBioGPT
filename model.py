from torch import optim
from torch import nn
from transformers import BioGptTokenizer, BioGptForCausalLM

from dataset import mimic_dataset
from settings import DEVICE


class BioGptForSequenceClassification(nn.Module):
    model_checkpoint: str = "microsoft/biogpt"
    optimizer: optim.Optimizer = optim.Adam()

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = BioGptTokenizer.from_pretrained(self.model_checkpoint)
        self.transformer = BioGptForCausalLM.from_pretrained(self.model_checkpoint)

    def forward(self, input_ids, labels):
        return self.transformer(input_ids=input_ids, labels=labels)

    def to(self):
        self.transformer.to(DEVICE)

    def generate(self, input_ids):
        prediction = self.transformer.generate(
            input_ids=input_ids,
            max_length=1024,
            min_length=5
        )
        # return prediction[:, 1:]
        return prediction

    def encode(self, prompt: str):
        output = self.tokenizer.encode(prompt, return_tensors="pt")
        return output

    def decode(self, encoder_output):
        output = self.tokenizer.decode(encoder_output[0], skip_special_tokens=True)
        return output


if __name__ == '__main__':
    example = mimic_dataset[0][0][0][:512]
    code = mimic_dataset[0][0][1]
    prompt = f"{example}\n\nThe relevant diagnosis ICD code is"
    model = BioGptForSequenceClassification()
    model(model.encode(prompt), model.encode(code))
