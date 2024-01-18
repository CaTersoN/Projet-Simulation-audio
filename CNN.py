#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:52:45 2023

@author: stani1
"""

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
from IPython.display import Audio, display
import matplotlib.pyplot as plt
import librosa
import sounddevice as sd


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


dossier_de_destination = 'C:/Users/hariz/Desktop/PJA/Projet-Simulation-audio/Spectro_amp'
dossier_de_destination_2 = 'C:/Users/hariz/Desktop/PJA/Projet-Simulation-audio/BDD_train'

liste_fichier=[]
liste_fichier_2=[]

# Charger les fichiers .npy
for fichier in os.listdir(dossier_de_destination):
    if fichier.endswith(".npy"):
        liste_fichier.append(fichier)

for fichier in os.listdir(dossier_de_destination_2):
    if fichier.endswith(".npy"):
        liste_fichier_2.append(fichier)

liste_de_spectro=[]
liste_de_spectro_dB=[]
liste_de_masque=[]

for fichier in liste_fichier :
    chemin_fichier = os.path.join(dossier_de_destination, fichier)
    liste_de_spectro.append(np.load(chemin_fichier))

for fichier in liste_fichier_2 :
    chemin_fichier = os.path.join(dossier_de_destination_2, fichier)
    if fichier[-5]=='m' :
        liste_de_masque.append(np.load(chemin_fichier))
    else : 
        liste_de_spectro_dB.append(np.load(chemin_fichier))

# input_data=np.real(liste_de_spectro[400:])
# target_masks=liste_de_masque

# dataset = SpeechDataset(input_data, target_masks)
# train_loader = DataLoader(dataset, batch_size=40, shuffle=True)


# num_epochs = 4
# loss_list=[]
# for epoch in range(num_epochs):
#     for inputs, masks in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, masks)
#         loss.backward()
#         optimizer.step()
#         loss_list.append(loss.item())
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
    
# torch.save(model.state_dict(), '/Users/stani1/Documents/GitHub/Projet-Simulation-audio/poids_du_modele.pth')
# torch.save(model.state_dict(), '/user/3/domers/traitement_parole/Projet-Simulation-audio/poids_du_modele.pth')
model.load_state_dict(torch.load('poids_du_modele_200.pth',map_location=torch.device('cpu')))
# model.load_state_dict(torch.load('/user/3/domers/traitement_parole/Projet-Simulation-audio/poids_du_modele.pth'))

model.eval()

def seuillage(M) :
    n,m=np.shape(M)
    m_s=np.zeros((n,m))
    for i in range (n) :
        for j in range (m) :
            if M[i,j]>0.48 :
                m_s[i,j]=1
    return m_s


def display_spectro_output(input1) :
    output=model(torch.from_numpy(input1).unsqueeze(0).float())
    output=output.detach().numpy()
    sr=16000
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(seuillage(output[0]), sr=sr, x_axis='time')
    plt.colorbar()
    plt.title('Mask')
    plt.show()


def display_spectro(input1) :
    sr=16000
    plt.figure(figsize=(5, 5))
    librosa.display.specshow(input1, sr=sr, x_axis='time',  y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogramme')
    plt.show()

def display_mask(input1) :
    sr=16000
    plt.figure(figsize=(3, 3))
    librosa.display.specshow(input1, sr=sr, x_axis='time', y_axis='linear')
    plt.title('Masque réel')
    plt.show()

def ecoute_denoise(i,sr) :
    output=model(torch.from_numpy(liste_de_spectro_dB[i]).unsqueeze(0).float())
    output=output.detach().numpy()
    mask=seuillage(output[0])
    print("ok")
    res= liste_de_spectro[i] * mask
    display_spectro(liste_de_spectro[i])
    display_spectro(res)
    ecoute(res, sr, i)
    print("okfin")

def ecoute(spectro,sr, i) :
    y_init = librosa.istft(liste_de_spectro[i], n_fft = 2048, hop_length = 512)
    plt.plot(y_init), plt.title("signal bruité"), plt.ylabel("Amplitude"), plt.xlabel("")
    plt.show()
    y=librosa.istft(spectro, n_fft = 2048, hop_length = 512)
    plt.plot(y), plt.title("signal débruité"), plt.ylabel("Amplitude")
    plt.show()
    sd.play(y, sr)
    sd.wait()

#display_mask(liste_de_masque[-1])
#print(len(liste_de_spectro))
#display_spectro(liste_de_spectro[0])

#ecoute(liste_de_spectro[600], 16000)
ecoute_denoise(300, 16000)

