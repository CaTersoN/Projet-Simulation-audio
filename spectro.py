# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import glob
import sounddevice as sd
import os
import random as rd

# Charger le fichier audio
noise_path = 'babble_16k.wav'
y_n, sr = librosa.load(noise_path,sr=16000)
B=y_n
speech_path='LibriSpeech/dev-clean/1993/147149/*.flac'

flac_list = glob.glob(speech_path)

def trouver_fichiers_flac(dossier):
    fichiers_flac = []
    for dossier_racine, sous_dossiers, fichiers in os.walk(dossier):
        for fichier in fichiers:
            if fichier.endswith(".flac"):
                chemin_fichier = os.path.join(dossier_racine, fichier)
                fichiers_flac.append(chemin_fichier)
    return fichiers_flac

# Remplacez '/chemin/du/dossier' par le chemin réel du dossier dans lequel vous souhaitez rechercher les fichiers .flac
dossier_a_explorer = 'LibriSpeech/dev-clean/'
fichiers_flac_trouves = trouver_fichiers_flac(dossier_a_explorer)


ex1=flac_list[5]
y_s, sr = librosa.load(ex1,sr=16000)

'''
# Calculer le spectrogramme
D_n = librosa.amplitude_to_db(librosa.stft(y_n[:np.size(y_s)]), ref=np.max)
y_ns=y_s+y_n[:np.size(y_s)]
D_s = librosa.amplitude_to_db(librosa.stft(y_s), ref=np.max)
D_ns = librosa.amplitude_to_db(librosa.stft(y_ns), ref=np.max)
# Afficher le spectrogramme
plt.figure(figsize=(10, 6))
librosa.display.specshow(D_ns, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogramme')
plt.show()
'''

def normalised_s(y_n,y_s,SNR) :
    d_n=len(y_n)
    d_s=len(y_s)
    if d_n>=d_s :
        i=rd.randint(0,d_n-d_s)
        y_n=y_n[i:i+d_s]
    else :
        i=rd.randint(0,d_s-d_n)
        y_s=y_s[i:i+d_n]
    A_n=np.max(y_n)-np.min(y_n)
    A_s=np.max(y_s)-np.min(y_s)
    y_n=y_n/A_n
    y_s=y_s/A_s
    A_s=10**(SNR/20)
    y_s=y_s*A_s
    return (y_n,y_s)

def mask (s_s,s_n) :
    n=len(s_s)
    m=len(s_s[0])
    M=np.zeros((n,m))
    for i in range (n) :
        for j in range (m) :
            if np.abs(s_s[i,j])**2>np.abs(s_n[i,j])**2 :
                M[i,j]=1
    return M

# =============================================================================
# Créer un jeu de n données
# =============================================================================
def create_training_set(Adr, n, SNR):
    Y=[]
    D=np.zeros(n)
    specTab = []
    y_ns = np.zeros(n)
    
    for i in range(n):
        Y.append(librosa.load(Adr[i], sr=16000)[0])
        
    if len(Y)<n:
        print("Taille de la liste inférieure à n")
        exit()
        
    for i in range(n):
        y_n, y_s = normalised_s(B,Y[i], SNR)
        y_ns = y_n + y_s
        D_n = librosa.stft(y_n, n_fft = 2048, hop_length = 512)
        D_s = librosa.stft(y_s, n_fft = 2048, hop_length = 512)
        D_ns = librosa.stft(y_ns, n_fft = 2048, hop_length = 512)
        S_n_dB = librosa.amplitude_to_db(np.abs(D_n),ref=np.max)
        S_s_dB = librosa.amplitude_to_db(np.abs(D_s),ref=np.max)
        S_ns_dB = librosa.amplitude_to_db(np.abs(D_ns),ref=np.max)
        specTab.append([S_ns_dB,S_s_dB,S_n_dB])
    return specTab

# =============================================================================
# Calcul des STFT
# =============================================================================

def calc_STFT(y_s,y_b):
    D1 = librosa.stft(y_s, n_fft = 2048, hop_length = 512)
    D2 = librosa.stft(y_b, n_fft = 2048, hop_length = 512)
    return D1, D2

def deci (i) :
    if i<10 :
        return '000'+str(i)
    if i<100 :
        return '00'+str(i)
    if i<1000 :
        return '0'+str(i)
    return str(i)
    
def ajusteTaille(Y, temps):
    n_ech_voulus = sr * temps
    for sig in Y:
        if len(sig) > n_ech_voulus:
            sig = sig[:n_ech_voulus]
        if len(sig) < n_ech_voulus:
            liste_zeros = np.zeros(n_ech_voulus - len(sig))
            sig = np.concatenate((sig, liste_zeros))

def list_SNR(n) : 
    l=np.linspace(-10,10,n)
    return l 
# =============================================================================
# Enregistrement des données
# =============================================================================    
n=10
SNR=20
liste_spectro=create_training_set(fichiers_flac_trouves, n, SNR)

dossier_de_destination = '/Users/stani1/Documents/Phelma/3A/Projet simulation logicielle/base de donnees/'

for i, fichier in enumerate(liste_spectro):
    nom_fichier = f"fichier_"+deci(i+1)+".npy"
    chemin_fichier = os.path.join(dossier_de_destination, nom_fichier)
    np.save(chemin_fichier, fichier)


liste_de_fichiers_charge = []
liste_fichier=[]
# Charger les fichiers .npy
for fichier in os.listdir(dossier_de_destination):
    if fichier.endswith(".npy"):
        chemin_fichier = os.path.join(dossier_de_destination, fichier)
        liste_fichier.append(fichier)
        
    
liste_fichier.sort()
for fichier in liste_fichier :
    chemin_fichier = os.path.join(dossier_de_destination, fichier)
    fichier_charge = np.load(chemin_fichier)
    liste_de_fichiers_charge.append(fichier_charge)


#sd.play(y_s, sr)
#sd.wait()
