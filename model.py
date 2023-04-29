from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from transformers import BioGptTokenizer, BioGptModel, BioGptPreTrainedModel, BioGptForCausalLM, \
    BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from dataset import mimic_dataset


class BioGptForSequenceClassification(BioGptPreTrainedModel):
    model_checkpoint: str = "microsoft/biogpt"

    def __init__(self, config) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.biogpt = BioGptModel(config=config)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # init weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        outputs = self.biogpt(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        pass

    def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
        pass

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        pass

    def prepare_inputs_for_generation(self, *args, **kwargs):
        pass


class BioGptForTest(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = BioGptForCausalLM.from_pretrained('microsoft/biogpt')

    def forward(self, input_ids, labels):
        return self.transformer(input_ids=input_ids, labels=labels)

    def to(self, device):
        self.transformer.to(device)

    def generate(self, input_tensor):
        max_length = 50
        min_length = 5

        prediction = self.transformer.generate(
            input_ids=input_tensor,
            num_beams=1,
            max_length=max_length + 1,
            min_length=min_length
        )

        return prediction[:, 1:]


if __name__ == '__main__':
    example = mimic_dataset[0][0][0][:512]
    code = mimic_dataset[0][0][1]
    prompt = f"{example}\n\nThe relevant diagnosis ICD code is"
    model = BioGptForSequenceClassification()
    model.tokenizer.encode(prompt)
    model(model.encode(prompt), model.encode(code))
