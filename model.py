import torch

from torch.nn import functional as F
from torch import nn
from transformers import BioGptModel, BioGptConfig

from dataset import mimic_loader
from preprocessing import TextProcessor
from settings import NUM_LABELS


class BioGptForSequenceClassification(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_labels = NUM_LABELS
        self.config = BioGptConfig()
        self.biogpt = BioGptModel(self.config)
        self.linear = nn.Linear(self.config.hidden_size, self.num_labels)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input_ids):
        outputs = self.biogpt(input_ids.squeeze())
        outputs = self.linear(outputs.last_hidden_state)
        outputs = self.softmax(outputs)
        return outputs

    def to(self, device):
        self.biogpt.to(device)


if __name__ == '__main__':
    # def _get_output():

    train_loader = mimic_loader.get('train')
    model = BioGptForSequenceClassification()
    for idx, item in enumerate(train_loader):
        input_ids, label_ids = item
        output = model(input_ids)
        prediction = torch.argmax(output, dim=-1)
        for out in prediction:
            natural_output = TextProcessor.decode(out)
        if idx == 0:
            break
