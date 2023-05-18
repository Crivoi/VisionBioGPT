import torch

from torch import nn
from transformers import BioGptModel, BioGptConfig, BioGptPreTrainedModel, BioGptForCausalLM

import settings
from dataset import mimic_loader
from preprocessing import TextProcessor
from settings.__init__ import NUM_LABELS

# class BioGptForLongDocumentClassification(BioGptPreTrainedModel):
#     def __init__(self, config) -> None:
#         super().__init__(config)
#         self.num_labels = NUM_LABELS
#         self.config = BioGptConfig()
#         self.biogpt = BioGptModel(self.config)
#         self.linear = nn.Linear(self.config.hidden_size, self.num_labels)
#
#     def forward(self, input_ids, labels=None, **kwargs):
#         outputs = self.biogpt(input_ids.squeeze(), )
#         outputs = self.linear(outputs.last_hidden_state)
#         if labels is not None:
#             loss_fct = nn.CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             return loss, logits
#         else:
#             return logits


# if __name__ == '__main__':
#     train_loader = mimic_loader.get('train')
#     model = BioGptForLongDocumentClassification()
#     for idx, item in enumerate(train_loader):
#         input_ids, label_ids = item
#         output = model(input_ids)
#         prediction = torch.argmax(output, dim=-1)
#         for out in prediction:
#             natural_output = TextProcessor.decode(out)
#             print(natural_output)
#         if idx == 0:
#             break
