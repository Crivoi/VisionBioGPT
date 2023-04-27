import torch
from transformers import TrainingArguments, Trainer

from dataset import train_loader, test_loader, mimic_dataset
from model import model

training_args: TrainingArguments = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=mimic_dataset.batch_size,
    per_device_eval_batch_size=mimic_dataset.batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer: Trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_loader,
    eval_dataset=test_loader,
    tokenizer=model.tokenizer,
)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

if __name__ == '__main__':
    trainer.train()
