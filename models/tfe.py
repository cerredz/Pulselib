import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import lightning as L
import numpy as np

class TFE(nn.Module):
    def __init__(self, dropout_rate:float=0.3):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.gelu = nn.GELU()

        # Fixed Sequential usage
        self.l1 = nn.Sequential(nn.ConvTranspose2d(1, 3, kernel_size=4, stride=2, padding=1), self.gelu)
        self.l2 = nn.Sequential(nn.ConvTranspose2d(3, 16, kernel_size=8, stride=2, padding=1), nn.BatchNorm2d(16), self.gelu)
        self.l3 = nn.Sequential(nn.ConvTranspose2d(16, 64, kernel_size=8, stride=2, padding=1), self.gelu, nn.Dropout2d(p=dropout_rate))  # Changed stride=3 to 2 for shape
        self.l4 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=(2,2), stride=2, padding=2), nn.BatchNorm2d(32), self.gelu, nn.Dropout2d(p=dropout_rate))
        self.l5 = nn.Sequential(nn.Conv2d(32, 16, kernel_size=(1,1), stride=1), self.gelu)
        self.l6 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=(1,1), stride=2), self.gelu, nn.Dropout2d(p=dropout_rate/2.0))
        self.l7 = nn.Sequential(nn.Linear(12*12*16, 384), self.gelu, nn.Dropout(p=dropout_rate/3.0))  # Adjusted flatten size after testing
        self.l8 = nn.Sequential(nn.Linear(384, 128), self.gelu, nn.Dropout(p=dropout_rate/3.0))
        self.l9 = nn.Sequential(nn.Linear(128, 16), self.gelu, nn.Dropout(p=dropout_rate/3.0))
        self.l10 = nn.Linear(16, 4)  # No Sequential, no GELU for output

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = x.view(x.size(0), -1)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        return x

class TFELightning(L.LightningModule):
    def __init__(self, lr: float=1e-3):
        super().__init__()
        self.lr = lr
        self.net = TFE()  # Embed the model
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        states, target_qs = batch
        preds = self(states)
        loss = self.loss_fn(preds, target_qs)
        self.log('train_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer