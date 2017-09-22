
# coding: utf-8

# In[99]:

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import datasets
import scipy.signal
import scipy.cluster.vq as vq
import scipy.io.wavfile as wav
import python_speech_features as speech
from sklearn import mixture
from sklearn.mixture import GaussianMixture as gmm
import scipy.spatial.distance as dist
import math


# In[216]:

def BIC(k, Sigs, Koefs, Lengths, penalty):
    P1 = Lengths[Koefs[k][0]]* np.log(np.cov(Sigs[Koefs[k][0]]))
    P2 = Lengths[Koefs[k][1]] * np.log(np.cov(Sigs[Koefs[k][1]]))
    P3 = (Lengths[Koefs[k][0]]+Lengths[Koefs[k][1]])*np.log(np.linalg.det(np.cov(Sigs[Koefs[k][0]], Sigs[Koefs[k][1]])))
    bic= 0.5*P1 + 0.5*P2-0.5*P3  + penalty*(2.5*(Lengths[Koefs[k][0]]+Lengths[Koefs[k][1]]))
    print(bic)
    return bic


# In[270]:

def fit_samples(samples, num):
	gmix = mixture.GaussianMixture(n_components=num, covariance_type='full')
	gmix.fit(samples)
	return (gmix.means_) #ovo su centroide

#glavna funkcija:
def calculate(Means, Mfccs, Sigs, Lengths, Freqs, M, num, penalty):
    N = min(num, 5)
    
    Koefs = []
    for i in range(0, N):
        Koefs.append([0,0])
    
    MaxVals = np.zeros((N))
    
    #racunanje N najblizih koeficijenata
    for i in range(0,num-1):
        for j in range(2,num):
            if(j>i):
                
                for l in range(0,N):
                    if(M[i][j]>=MaxVals[l]):
                        for k in range(N, l+1):
                            MaxVals[k]=MaxVals[k-1]
                            Koefs[k] = Koefs[k-1]
                        MaxVals[l] = M[i][j]
                        Koefs[l] = [i,j]
                        break
                 
    #racunanje najveceg BIC
    Bics = []
        
    for i in range(0, N):
        Bics.append(BIC(i, Sigs, Koefs, Lengths, penalty))
    
    max_bic = Bics[0]
    max_i=0
    for i in range(1,N):
        if(Bics[i]>max_bic):
            max_bic = Bics[i]
            max_i = i
        
    Means_new = []
    Mfccs_new = []
    Sigs_new = []
    Lengths_new = []
    Freqs_new = []
    M_new = np.zeros((num, num))
    
    if(max_bic<0):
        return (Means_new, Mfccs_new, Sigs, Lengths_new, Freqs_new, M_new, -1)
    
    [i,j] = Koefs[max_i]
    spoji_klastere(i, j)
    sig_new = 0.5*Sigs[i]+0.5*Sigs[j]
    means_new = 0.5*Means[i]+0.5*Means[j]
    
    for k in range (0,num):
        if(k!=i and k!=j):
            Sigs_new.append(Sigs[k])
            Mfccs_new.append(Mfccs[k])
            Means_new.append(Means[k])
            Lengths_new.append(Mfccs[k].shape[0])
            Freqs_new.append(Freqs[k])
    
    for k in range(0, num-1):
        if(k<i):
            for l in range(0, num-1):
                if(l<i):
                    M_new[k][l]=M[k][l]
                else:
                    M_new[k][l]=M[k][l+1]
        else:
            for l in range(0, num-1):
                if(l<i):
                    M_new[k][l]=M[k+1][l]
                else:
                    M_new[k][l]=M[k+1][l+1]
                    
    for k in range(0, num-1):
        if(k<j):
            for l in range(0, num-1):
                if(l<j):
                    M_new[k][l]=M[k][l]
                else:
                    M_new[k][l]=M[k][l+1]
        else:
            for l in range(0, num-1):
                if(l<j):
                    M_new[k][l]=M[k+1][l]
                else:
                    M_new[k][l]=M[k+1][l+1]

    freq = means_new
    freq_sum = np.sum(freq)
    freq = (freq_sum - freq)/freq_sum
    
    for k in range(0, num-2):
        M_new[num-2][k] = np.abs(dist.cosine(Freqs[k], freq))
               
    Lengths_new.append(speech.mfcc(sig_new).shape[0])
    Means_new.append(means_new)        
    Mfccs_new.append(speech.mfcc(sig_new))
    Sigs_new.append(sig_new)
    Freqs_new.append(freq)
    return (Means_new, Mfccs_new, Sigs_new, Lengths_new, Freqs_new, M_new, 1)


# In[281]:

def spoji_klastere(i, j):   
    clust_1 = Clusters_True[i]
    clust_2 = Clusters_True[j]
    
    for k in range(0, len(Clusters_Pred)):
        if(Clusters_Pred[k]==clust_2):
            Clusters_Pred[k]=clust_1


# In[282]:

(rate,sig1) = wav.read("richard3.wav")
(rate,sig2) = wav.read("amy3.wav")
(rate,sig3) = wav.read("derek3.wav")
(rate,sig4) = wav.read("paolo3.wav")
(rate,sig5) = wav.read("nilofer3.wav")

People =[sig1, sig2, sig3, sig4, sig5]
Sigs=[]

Clusters_True = []
Clusters_Pred = []

k=1
c = 0
for sig in People:
    while((2000*k)<sig.shape[0]):
        sig1 = sig[(k-1)*2000: k*2000]
        Sigs.append(sig1[:,0])
        Clusters_True.append(c)
        k = k+1
    c = c+1

for i in (0, len(Clusters_True)):
    Clusters_Pred.append(i)
    
num = len(Sigs)
br_klast = 0
        
mfcc_feat = np.copy(speech.mfcc(Sigs[0]))
Lengths = []#mfcc lengths
Mfccs = []
Freqs = []
    
for i in range(0, num):
    mfcc = speech.mfcc(Sigs[i])
    Mfccs.append(mfcc)
    if(i>0):
         mfcc_feat = np.vstack((mfcc_feat, speech.mfcc(Sigs[i])))
    Lengths.append(mfcc.shape[0])   
    
Means = fit_samples(mfcc_feat, num)

for i in range(0, num):
    freq = Means[i]
    freq_sum = np.sum(freq)
    freq = (freq_sum - freq)/freq_sum
    Freqs.append(freq)
    
M = np.zeros((num,num))
for i in range(0,num):
    for j in range(0,num):
        if(i==j):
            M[i][j]=0
        else:
            M[i][j] = np.abs(dist.cosine(Freqs[i], Freqs[j]))
        


# In[283]:

ind = 1
penalty = 3.3

while(ind>0 and num>1):
    (Means, Mfccs, Sigs, Lengths, Freqs, M, ind) = calculate(Means, Mfccs, Sigs, Lengths, Freqs, M, num, penalty)
    num = num-1

for i in range(0, len(Clusters_Pred)):
    Clusters_Pred[i] = i
    
print(Clusters_Pred)


# In[ ]:



