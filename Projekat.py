
# coding: utf-8

# In[52]:

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


# In[53]:

(rate,sig1) = wav.read("eric.wav")
(rate,sig2) = wav.read("amy.wav")

mfcc1 = speech.mfcc(sig1)
mfcc2 = speech.mfcc(sig2)

def fit_samples(samples):
	gmix = mixture.GaussianMixture(n_components=2, covariance_type='full')
	gmix.fit(samples)
	return (gmix.means_) #ovo su centroide
    
mfcc_feat=np.vstack((mfcc1, mfcc2))

gmms = fit_samples(mfcc_feat)

mean1 = np.mean(mfcc1)
cov_mat1 = np.cov(mfcc1)

mean2 = np.mean(mfcc2)
cov_mat2 = np.cov(mfcc2)

vq.whiten(mfcc_feat)
kmeans = vq.kmeans(obs=mfcc_feat, k_or_guess=gmms)
freq1 = kmeans[0][0] #codebook
freq_sum1 = np.sum(freq1)
freq1 = (freq_sum1 - freq1)/freq_sum1
freq2 = kmeans[0][1] #codebook
freq_sum2 = np.sum(freq2)
freq2 = (freq_sum2 - freq2)/freq_sum2

print(dist.cosine(freq1, freq2))


# In[ ]:



