from typing import Union, Tuple, Optional

import torch
from torch import nn
from transformers import BioGptForCausalLM, BioGptForSequenceClassification, BioGptTokenizer
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

import settings


class BioGptTestModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.biogpt = BioGptForSequenceClassification.from_pretrained(settings.BIOGPT_CHECKPOINT,
                                                                      num_labels=settings.NUM_LABELS,
                                                                      problem_type="multi_label_classification")

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        outputs = self.biogpt(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels
        )
        return outputs


if __name__ == '__main__':
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    model = BioGptForSequenceClassification.from_pretrained("microsoft/biogpt",
                                                            problem_type="multi_label_classification")

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]
    print(predicted_class_ids)
    # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
    # num_labels = len(model.config.id2label)
    # model = BioGptForSequenceClassification.from_pretrained(
    #     "microsoft/biogpt", num_labels=num_labels, problem_type="multi_label_classification"
    # )
    #
    # labels = torch.sum(
    #     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
    # ).to(torch.float)
    # loss = model(**inputs, labels=labels).loss
