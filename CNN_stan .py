#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:51:49 2023

@author: stani1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class MaskingNetwork(nn.Module):
    def __init__(self):
        super(MaskingNetwork, self).__init__()
        
        # Couches de convolution
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        
        # Couches entièrement connectées
        self.fc1 = nn.Linear(64 * (taille_spectrogramme//4) * (taille_spectrogramme//4), 128)
        self.fc2 = nn.Linear(128, taille_spectrogramme * taille_spectrogramme)  # Output: masque binaire de même taille que le spectrogramme
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * (taille_spectrogramme//4) * (taille_spectrogramme//4))
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Utilisation de sigmoid pour obtenir des valeurs entre 0 et 1 pour le masque
        return x

criterion = nn.BCELoss()


import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeechEnhancementModel(nn.Module):
    def __init__(self):
        super(SpeechEnhancementModel, self).__init__()

        # Encodeur
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Décodeur
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Utilisation de sigmoid pour obtenir des valeurs entre 0 et 1 pour le masque
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Supposons que vous avez un DataLoader pour votre base de données
# train_loader = ...

model = SpeechEnhancementModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    for inputs, masks in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


import torch
from torch.utils.data import Dataset, DataLoader

class SpeechDataset(Dataset):
    def __init__(self, input_data, target_masks):
        self.input_data = input_data
        self.target_masks = target_masks

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        input_spec = torch.from_numpy(self.input_data[index]).unsqueeze(0).float()
        target_mask = torch.from_numpy(self.target_masks[index]).unsqueeze(0).float()

        return input_spec, target_mask

# Exemple d'utilisation
# input_data et target_masks sont vos données et masques binaires respectivement

dataset = SpeechDataset(input_data, target_masks)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)


