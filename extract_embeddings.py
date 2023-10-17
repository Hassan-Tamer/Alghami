from torchvggish import vggish, vggish_input
import numpy as np
import os

# Initialise model and download weights
embedding_model = vggish()
embedding_model.eval()

path = "Dataset/genres_original/"
genres = os.listdir(path)
for genre in genres:
    os.mkdir(genre)
    full_path = path + genre
    songs = os.listdir(full_path)
    for song in songs:
        try:
            example = vggish_input.wavfile_to_examples(full_path + "/" + song)
            embeddings = embedding_model.forward(example)
            embeddings = embeddings.detach().numpy()
            np.save(genre + "/" + song, embeddings)
        except:
            print("Error with song: " + song)
            continue