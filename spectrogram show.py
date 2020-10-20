from matplotlib import pyplot as plt
import librosa.display
import numpy as np
from matplotlib import pyplot as plt
import math
import librosa.display
from scipy import signal

#zxx = np.loadtxt('/home/shaoyu/Downloads/Keras-GAN-master/pix2pix/datasets/fabric1/train/8/generated mask/200.txt')
zxx = np.loadtxt('/home/shaoyu/Desktop/FrictGAN/New Folder/pre_spec.txt')
#zxx = np.loadtxt('/home/shaoyu/Downloads/Keras-GAN-master/pix2pix/datasets/fabric1/train/2/output_data/135.txt')
a = zxx
print(zxx.shape)
print(zxx.mean())
print(zxx.max())
print(zxx.min())
print(zxx.mean())
print(zxx.max())
print(zxx.min())
y_inv1 = np.abs(librosa.core.griffinlim(zxx))
print(zxx.shape)
#plt.show()
plt.figure()
#plt.figure(figsize=(1.28, 1.28), dpi=300)
#librosa.display.specshow(librosa.amplitude_to_db(zxx, ref=np.max), y_axis='log', x_axis='time') #log for frequency
librosa.display.specshow(librosa.amplitude_to_db(a, ref=np.max))  # No log for frequency
#plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)
plt.show()
"""
S = np.loadtxt('pre_spec.txt')
T = np.loadtxt('tar.txt')
y_inv = np.abs(librosa.core.griffinlim(S, hop_length=128, win_length=512, window='hamming', center=True, length=None, init=None))
y_inv1 = np.abs(librosa.core.griffinlim(T, hop_length=128, win_length=512, window='hamming', center=True, length=None, init=None))
n = 1280
x = range(0,n)
ax = plt.gca()
ax.plot(x[:1280], y_inv[:1280])
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)
#ax.set_xlabel('Displacement')
#plt.title('Generated Signals')
plt.show()
"""
S = np.loadtxt('pre_spec.txt')
T = np.loadtxt('tar.txt')
y_inv = np.abs(librosa.core.griffinlim(S, hop_length=128, win_length=512, window='hamming', center=True, length=None, init=None))
y_inv1 = np.abs(librosa.core.griffinlim(T, hop_length=128, win_length=512, window='hamming', center=True, length=None, init=None))
print('.............................................................')
print(y_inv)
print('.............................................................')
print(y_inv1)
print('.............................................................')
err = np.sum((y_inv - y_inv1) ** 2)
print(err/1280.0)

plt.figure()
plt.subplot(2, 1, 1)
# librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time') #log for frequency
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max))  # No log for frequency
plt.title('Generated Signals')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()

plt.subplot(2, 1, 2)
# librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time') #log for frequency
librosa.display.specshow(librosa.amplitude_to_db(T, ref=np.max))  # No log for frequency
plt.title('Original Signals')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

n = 1280
x = range(0,n)
b, a = signal.butter(8, 0.02, 'lowpass')
filte_y_inv = signal.filtfilt(b, a, y_inv)
filte_y_inv1 = signal.filtfilt(b, a, y_inv1)

plt.subplot(2, 1, 1)
ax = plt.gca()
ax.plot(x[:1280], y_inv[:1280])
#ax.set_xlabel('Displacement')
#plt.title('Generated Signals')

plt.subplot(2, 1, 2)
ax = plt.gca()
ax.plot(x[:1280], y_inv1[:1280])
#ax.set_xlabel('Displacement')

#plt.title('Original Signals')

plt.subplot(4, 1, 3)
ax = plt.gca()
ax.plot(x[:1280], filte_y_inv[:1280])

plt.subplot(4, 1, 4)
ax = plt.gca()
ax.plot(x[:1280], filte_y_inv1[:1280])

plt.show()
