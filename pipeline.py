import torch
from torch import nn
from tqdm import tqdm
from transformers import TrainingArguments, Trainer, BioGptForCausalLM, BioGptTokenizer

import settings
from dataset import mimic_loader
from model import BioGptForTest


# from model import model

# training_args: TrainingArguments = TrainingArguments(
#     output_dir='./results',
#     learning_rate=2e-5,
#     per_device_train_batch_size=mimic_dataset.batch_size,
#     per_device_eval_batch_size=mimic_dataset.batch_size,
#     num_train_epochs=5,
#     weight_decay=0.01,
# )
#
# trainer: Trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_loader,
#     eval_dataset=test_loader,
#     tokenizer=model.tokenizer,
# )


class BioGptTrainer:
    def __init__(self, model, train_loader, lr=2e-4):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = settings.DEVICE

    def train(self):
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_steps = 0
        for batch in tqdm(self.train_loader):
            # Move batch to device
            # batch = {k: v.to(device) for k, v in batch.items()}

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = model(batch[0].squeeze(), batch[1].squeeze())
            loss = self.criterion(torch.mean(outputs, dim=2), batch[1].squeeze().float())

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Compute accuracy
            preds = torch.argmax(outputs, dim=2)
            accuracy = torch.mean((preds == batch[1].squeeze()).float())

            # Update metrics
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            total_steps += 1

        avg_loss = total_loss / total_steps
        avg_accuracy = total_accuracy / total_steps
        return avg_loss, avg_accuracy


if __name__ == '__main__':
    model = BioGptForTest()
    trainer = BioGptTrainer(model=model, train_loader=mimic_loader.get('train'))
    trainer.train()
