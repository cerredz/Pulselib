import os
import torch
from torch import nn
from torch.utils.datasets import DataLoader
import lightning as L
import numpy as np

class Agent(nn.module):
    def __init__(self, dropout_rate:int=.3):
        super().__init__()
        self.dropout_rate=dropout_rate
        self.gelu=nn.GELU()

        # start off upsampling the 4x4x1 board, "same" padding
        self.l1 = nn.Sequential(nn.ConvTranspose2d(1,3, kernel_size=4, stride=2, padding=1), self.gelu())
        self.l2 = nn.Sequential(nn.ConvTranspose2d(3, 16, kernel_size=8, stride=2, padding=1), nn.BatchNorm2d(16), self.gelu())
        self.l3 = nn.Sequential(nn.ConvTranspose2d(16, 64, kernel_size=8, stride=3, padding=1), self.gelu(), nn.Dropout2d(p=dropout_rate))
        self.l4 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=(2,2), stride=2, padding=2), nn.BatchNorn2d(32), self.gelu, nn.Dropout2d(p=dropout_rate))
        self.l5 = nn.Sequential(nn.Conv2d(32, 16, kernel_size=(1,1), stride=1), self.gelu())
        self.l6 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=(1,1), stride=2), self.gelu(), nn.Dropout2d(p=dropout_rate/2.0))
        self.l7 = nn.Sequential(nn.Linear(8*8*16, 384), self.gelu(), nn.Dropout2d(p=dropout_rate/3.0))
        self.l8 = nn.Sequential(nn.Linear(384, 128), self.gelu(), nn.Dropout2d(p=dropout_rate/3.0))
        self.l9 = nn.Sequential(nn.Linear(128, 16), self.gelu(), nn.Dropout2d(p=dropout_rate/3.0))
        self.l10 = nn.Sequential(nn.Linear(16, 4), self.gelu())

    def forward(self, x):
        x=self.l1(x)
        x=self.l2(x)
        x=self.l3(x)
        x=self.l4(x)
        x=self.l5(x)
        x=self.l6(x)
        x=np.flatten(x)
        x=self.l7(x)
        x=self.l8(x)
        x=self.l9(x)
        x=self.l10(x)
        return x
        
class AgentLightning(L.LightningModule):
    def __init__(self, agent, lr: float=1e-3):
        super().__init__()
        self.agent=agent
        self.lr=lr

    def training_step(self, batch, batch_idx):
        x, _ = batch
        pass

    def configure_optimizer(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer