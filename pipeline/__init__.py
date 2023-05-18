from typing import Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BioGptModel, get_scheduler, BioGptTokenizer
from torch.optim import AdamW

import settings
from dataset import MimicDataset, build_dataloader, Collator
from model import BioGptTestModel


class Trainer:
    model: nn.Module
    num_epochs: int = 1
    optimizer: Optimizer

    def __init__(
            self,
            model: BioGptTestModel,
            train_loader: DataLoader,
            dev_loader: DataLoader,
            label2idx: Dict
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.label2idx = label2idx
        self.idx2label = {v: k for k, v in self.label2idx.items()}
        self.num_training_steps = self.num_epochs * len(self.train_loader)
        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_training_steps
        )
        self.progress_bar = tqdm(range(self.num_training_steps))

    def train_iteration(self, input_ids, labels, attention_mask):
        self.model.zero_grad()

        output = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

        loss = output.loss
        loss.backward()

        self.optimizer.step()
        self.lr_scheduler.step()
        self.progress_bar.update(1)

        return loss.item()

    def train(self, plot=True):
        self.model.train()

        plot_loss_total = 0
        plot_losses = []
        for epoch in range(self.num_epochs):
            for idx, batch in enumerate(self.train_loader):
                batch = {k: v.to(settings.DEVICE) for k, v in batch.items()}
                loss = self.train_iteration(**batch)
                plot_loss_total += loss
                plot_loss_avg = plot_loss_total / 100
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        if plot:
            plt.figure()
            fig, ax = plt.subplots()
            ax.plot(plot_losses)
            plt.show()

    def evaluate(self):
        self.model.eval()

        n_correct = []
        i = 0
        with torch.no_grad():
            for batch in tqdm(self.dev_loader, desc="Evaluating"):
                logits = self.model(**batch).logits

                # preds = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]
                preds = torch.arange(0, logits.shape[0] * logits.shape[1]).reshape(logits.shape)[
                    torch.sigmoid(logits).squeeze(dim=0) > 0.5].reshape(8, -1)
                ground_truth = batch['labels']
                if i < 10:
                    print('\n')
                    print(preds)
                    # print([self.idx2label[x.item()] for x in preds])
                    print(ground_truth)
                    print(preds.shape, ground_truth.shape)
                    i += 1
                else:
                    break

                n_correct.append(np.all(preds == ground_truth))
        accuracy = np.mean(n_correct)

        self.model.train()
        return accuracy
