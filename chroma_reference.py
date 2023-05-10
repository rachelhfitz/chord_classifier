import matplotlib.pyplot as plt
import librosa
import numpy as np


x, sr = librosa.load('data/training_set/home_piano/C-4-minor_0.wav')

# full chromagram
chromagram = librosa.feature.chroma_stft(y=x, sr=sr)
plt.figure(figsize=(10, 4))
librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram')
plt.tight_layout()
plt.show()

# mean of each bin
mean_chroma = np.mean(chromagram, axis=1)
fig = plt.figure(figsize = (10,5))
plt.bar(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'], mean_chroma)
plt.xlabel("Notes")
plt.ylabel("Mean Chromagram")
plt.show()