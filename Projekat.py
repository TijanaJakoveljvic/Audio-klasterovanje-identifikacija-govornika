import numpy as np
import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt
from sklearn import datasets
import scipy.cluster.vq as vq
import scipy.io.wavfile as wav
import python_speech_features as speech
from sklearn import mixture
from sklearn.mixture import GaussianMixture as gmm
import scipy.spatial.distance as dist
import glob
import python_speech_features as speech
from scipy.io.wavfile import read
# np.set_printoptions(threshold=np.inf)

number_of_different_speakers = 2

audio_segments = []
mfcc_data = []
for filename in glob.glob('*.wav'):
	(rate, data) = read(filename)
	audio_segments.append(data)
	mfcc_data.append(speech.mfcc(data))

mfcc = mfcc_data[0]
for x in range(1, len(mfcc_data)):
	mfcc = np.vstack((mfcc, mfcc_data[x]))

def calculate_centroids(samples):
    gmix = mixture.GaussianMixture(n_components=number_of_different_speakers, covariance_type='full')
    gmix.fit(samples)
    return (gmix.means_)

gmms = calculate_centroids(mfcc)

mean = []
cov_mat = []
for item in mfcc_data:
    mean_i = np.mean(item)
    mean.append(mean_i)
    cov_i = np.cov(item)
    cov_mat.append(np.cov(item))

whitened = vq.whiten(mfcc)
(codebook, distortion) = vq.kmeans(obs=whitened, k_or_guess=gmms)
code = vq.vq(whitened, codebook)

frequency = []
for freq in codebook:
	freq_sum = np.sum(freq)
	freq = (freq_sum - freq)/freq_sum
	frequency.append(freq)

# print(dist.cosine(freq1, freq2))
