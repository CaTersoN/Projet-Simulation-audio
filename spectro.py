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
noise_path = 'C:/Users/hariz/Desktop/PJA/babble_16k.wav'
y_n, sr = librosa.load(noise_path,sr=16000)
B=y_n

def trouver_fichiers_flac(dossier):
    fichiers_flac = []
    for dossier_racine, sous_dossiers, fichiers in os.walk(dossier):
        for fichier in fichiers:
            if fichier.endswith(".flac"):
                chemin_fichier = os.path.join(dossier_racine, fichier)
                fichiers_flac.append(chemin_fichier)
    return fichiers_flac

dossier_a_explorer = 'C:/Users/hariz/Desktop/PJA/LibriSpeech/dev-clean/'
fichiers_flac_trouves = trouver_fichiers_flac(dossier_a_explorer)

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
    y_n = y_n / A_n
    y_s = y_s / A_s
    A=10**(SNR/20)
    y_n=y_n/A
    return (y_n,y_s)

def mask (s_s,s_n) :
    n=len(s_s)
    m=len(s_s[0])
    M=np.zeros((n,m))
    for i in range (n) :
        for j in range (m) :
            if np.abs(s_s[i][j])**2>np.abs(s_n[i][j])**2 :
                M[i,j]=1
    return M

# =============================================================================
# Créer un jeu de n données
# =============================================================================

def create_training_set(Adr, n, SNR, temps, sr):
    Y=[]
    D=np.zeros(n)
    specTab = []
    y_ns = np.zeros(n)
    for i in range(n):
        Y.append(librosa.load(Adr[i], sr=16000)[0])
    Y=ajusteTaille(Y, temps, sr)
    if len(Y)<n:
        print("Taille de la liste inférieure à n")
        exit()
    for i in range(n):
        y_n, y_s = normalised_s(B,Y[i], SNR[i])
        y_ns = y_n + y_s
        D_s = librosa.stft(y_s, n_fft = 2048, hop_length = 512)
        D_n = librosa.stft(y_n, n_fft = 2048, hop_length = 512)
        D_ns = librosa.stft(y_ns, n_fft = 2048, hop_length = 512)
        S_ns_dB = librosa.amplitude_to_db(np.abs(D_ns),ref=np.max)
        specTab.append([D_ns])
    return specTab

def create_test_set(Adr, n, SNR, temps, sr):
    Y=[]
    specTab = []
    y_ns = np.zeros(n)
    for i in range(n):
        Y.append(librosa.load(Adr[-i], sr=16000)[0])
    Y=ajusteTaille(Y, temps, sr)
    if len(Y)<n:
        print("Taille de la liste inférieure à n")
        exit()
    for i in range(n):
        y_n, y_s = normalised_s(B,Y[i], SNR[i])
        y_ns = y_n + y_s
        D_s = librosa.stft(y_s, n_fft = 2048, hop_length = 512)
        D_n = librosa.stft(y_n, n_fft = 2048, hop_length = 512)
        D_ns = librosa.stft(y_ns, n_fft = 2048, hop_length = 512)
        S_ns_dB = librosa.amplitude_to_db(np.abs(D_ns),ref=np.max)
        specTab.append([S_ns_dB, D_s, D_ns])
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
    
def ajusteTaille(Y, temps, sr):
    n_ech_voulus = sr * temps
    Y_n=[]
    
    
    for sig in Y:
        
        if len(sig) > n_ech_voulus:
            sig_2 = sig[:n_ech_voulus]
            Y_n.append(sig_2)
            
            
        if len(sig) <= n_ech_voulus:
            liste_zeros = np.zeros(n_ech_voulus - len(sig))
            sig_2 = np.concatenate((sig, liste_zeros))
            Y_n.append(sig_2)
          
    return Y_n
    
def list_SNR(n) : 
    l=np.linspace(-10,10,n)
    return l 
# =============================================================================
# Enregistrement des données
# =============================================================================    
print('TEST')
n=1200
SNR=list_SNR(n)
liste_spectro=create_training_set(fichiers_flac_trouves, n, SNR, 3, sr)
#liste_spectro=create_test_set(fichiers_flac_trouves, n, SNR, 3, sr)
dossier_de_destination = 'C:/Users/hariz/Desktop/PJA/Projet-Simulation-audio/Spectro_amp'


# for i, fichier in enumerate(liste_spectro):
#     nom_fichier = f"fichier_"+deci(i+1)+".npy"
#     nom_fichier_m = f"fichier_"+deci(i+1)+"_m"+".npy"
#     chemin_fichier = os.path.join(dossier_de_destination, nom_fichier)
#     chemin_fichier_m = os.path.join(dossier_de_destination, nom_fichier_m)
#     fichier_m=mask(liste_spectro[i][1],liste_spectro[i][2])
    
#     np.save(chemin_fichier, fichier[0])
#     np.save(chemin_fichier_m, fichier_m)

for i, fichier in enumerate(liste_spectro):
    nom_fichier = f"spectro_"+deci(i+1)+".npy"
    chemin_fichier = os.path.join(dossier_de_destination, nom_fichier)
    np.save(chemin_fichier, fichier[0])
