from transformers import TrainingArguments, Trainer

from dataset import train_loader, test_loader, mimic_dataset
from model import model, tokenizer

training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=mimic_dataset.batch_size,
    per_device_eval_batch_size=mimic_dataset.batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_loader,
    eval_dataset=test_loader,
    tokenizer=tokenizer,
)

trainer.train()

if __name__ == '__main__':
    pass
