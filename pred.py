from torchvggish import vggish, vggish_input
import torch

mod = torch.load("VGGish/best.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


embedding_model = vggish()
embedding_model.eval()

pred_path = "/home/hassan/Documents/eme/Alghami/Dataset/genres_original/disco/disco.00000.wav"
try:
    example = vggish_input.wavfile_to_examples(pred_path)
    embeddings = embedding_model.forward(example)
    embeddings = embeddings.detach().numpy()
    embeddings/=255.0
    embeddings = embeddings.reshape(1,embeddings.shape[0]*embeddings.shape[1])
    embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
except:
    print("Error in extracting embedding")
    exit(1)


out = mod(embeddings)
pred = (torch.max(torch.exp(out), 1)[1]).data.cpu().numpy()
classes = ("blues" , "classical" , "country" , "disco" , "hiphop" , "jazz" , "metal" , "pop" , "reggae" , "rock")
print(classes[pred[0]])