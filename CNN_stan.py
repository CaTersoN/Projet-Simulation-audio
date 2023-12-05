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
import os
import numpy as np
import matplotlib.pyplot as plt
#import librosa


class SpeechEnhancementModel(nn.Module):
    def __init__(self):
        super(SpeechEnhancementModel, self).__init__()

        # Encodeur
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Décodeur
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Sigmoid()  # Utilisation de sigmoid pour obtenir des valeurs entre 0 et 1 pour le masque
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

criterion = nn.BCELoss()

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



model = SpeechEnhancementModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# dossier_de_destination = '/Users/stani1/Documents/GitHub/Projet-Simulation-audio/base de donnees_10/'
dossier_de_destination = '/user/3/domers/traitement_parole/Projet-Simulation-audio/0_100/'


liste_de_fichiers_charge = []
liste_fichier=[]
liste_nom=[]
# Charger les fichiers .npy
for fichier in os.listdir(dossier_de_destination):
    if fichier.endswith(".npy"):
        chemin_fichier = os.path.join(dossier_de_destination, fichier)
        liste_nom.append(chemin_fichier)
        liste_fichier.append(fichier)
        

liste_fichier.sort()
liste_nom.sort()
liste_de_spectro=[]
liste_de_masque=[]
for fichier in liste_fichier :
    chemin_fichier = os.path.join(dossier_de_destination, fichier)
    #print(fichier)
    if fichier[-5]=='m' :
        liste_de_masque.append(np.load(chemin_fichier))
    else : 
        liste_de_spectro.append(np.load(chemin_fichier)[0])
   
input_data=np.real(liste_de_spectro)
target_masks=liste_de_masque

dataset = SpeechDataset(input_data, target_masks)
train_loader = DataLoader(dataset, batch_size=40, shuffle=True)

num_epochs = 50
loss_list=[]
for epoch in range(num_epochs):
    for inputs, masks in train_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
    
# torch.save(model.state_dict(), '/Users/stani1/Documents/GitHub/Projet-Simulation-audio/poids_du_modele.pth')
torch.save(model.state_dict(), '/user/3/domers/traitement_parole/Projet-Simulation-audio/poids_du_modele.pth')
# model.load_state_dict(torch.load('/Users/stani1/Documents/GitHub/Projet-Simulation-audio/poids_du_modele.pth'))
model.load_state_dict(torch.load('/user/3/domers/traitement_parole/Projet-Simulation-audio/poids_du_modele.pth'))

model.eval()


# def display_spectro_output(input1) :
#     output=model(torch.from_numpy(input1).unsqueeze(0).float())
#     output=output.detach().numpy()
#     sr=16000
#     plt.figure(figsize=(10, 6))
#     librosa.display.specshow(output[0], sr=sr, x_axis='time')
#     plt.colorbar()
#     plt.title('Mask')
#     plt.show()


# def display_spectro(input1) :
#     sr=16000
#     plt.figure(figsize=(10, 6))
#     librosa.display.specshow(input1, sr=sr, x_axis='time')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Spectrogramme')
#     plt.show()

