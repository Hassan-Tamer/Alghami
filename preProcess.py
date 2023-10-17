import numpy as np
import os
from natsort import natsorted


X = []
y = []

path = "VGGish/Embeddings/"
genres = natsorted(os.listdir(path))

for i,genre in enumerate(genres):
    full_path = path + genre
    songs = os.listdir(full_path)
    for song in songs:
        try:
            embeddings = np.load(full_path + "/" + song)
            y.append(i)
            X.append(embeddings)
        except Exception as e:
            print(e)
            continue
    print("Finished genre: " + genre)

# Convert the list of tuples to a NumPy array
X = np.array(X)
y = np.array(y)

np.save("VGGish/X.npy", X)
np.save("VGGish/y.npy", y)

print("SAVED")