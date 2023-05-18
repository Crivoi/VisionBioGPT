import torch
from transformers import AutoTokenizer, BioGptForSequenceClassification

import settings
from dataset import MimicDataset, Collator, build_dataloader
from settings.utils import Splits

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
    model = BioGptForSequenceClassification.from_pretrained("microsoft/biogpt", num_labels=settings.NUM_LABELS,
                                                            problem_type="multi_label_classification")

    train_dataset = MimicDataset(tokenizer=tokenizer,
                                 split=Splits.train.value,
                                 cache_dir='./cache-temp')

    # dev_dataset = MimicDataset(tokenizer=tokenizer, split=Splits.dev.value, label2idx=train_dataset.label2idx,
    #                            cache_dir='./cache-small')

    # data_collator = Collator(tokenizer=tokenizer, max_seq_length=settings.MAX_SEQ_LENGTH)
    # train_loader = build_dataloader(train_dataset, data_collator)
    # dev_loader = build_dataloader(dev_dataset, data_collator)

    idx2label = {v: k for k, v in train_dataset.label2idx.items()}

    inputs = train_dataset[0].get('input_ids').reshape(1, -1)
    labels = train_dataset[0].get('labels').reshape(1, -1)

    with torch.no_grad():
        logits = model(inputs).logits

    predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]
    preds = [idx2label[x.item()] for x in predicted_class_ids]
    # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
    num_labels = settings.NUM_LABELS
    loss = model(inputs, labels=labels.float()).loss
    print(loss)
