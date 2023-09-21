import numpy as np
import os

X = []
y = []

path = "Embeddings/"
genres = os.listdir(path)
for genre in genres:
    full_path = path + genre
    songs = os.listdir(full_path)
    for song in songs:
        try:
            embeddings = np.load(full_path + "/" + song)
            y.append(genre)
            X.append(embeddings)
        except Exception as e:
            print(e)
            continue
    print("Finished genre: " + genre)

# Convert the list of tuples to a NumPy array
X = np.array(X)
y = np.array(y)

np.save("X.npy", X)
np.save("y.npy", y)
