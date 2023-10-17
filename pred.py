from torchvggish import vggish, vggish_input
import torch
import torch.nn.functional as F
import streamlit as st
import wave
from pydub import AudioSegment

class NN_model(torch.nn.Module):
        def _init_(self,input_size,hidden1_size,hidden2_size,hidden3_size,hidden4_size,hidden5_size,output_size):
            super(NN_model,self)._init_()
            self.fc1 = torch.nn.Linear(input_size, hidden1_size)
            self.fc2 = torch.nn.Linear(hidden1_size, hidden2_size)
            self.fc3 = torch.nn.Linear(hidden2_size, hidden3_size)
            self.fc4 = torch.nn.Linear(hidden3_size, hidden4_size)
            self.fc5 = torch.nn.Linear(hidden4_size, hidden5_size)
            self.fc6 = torch.nn.Linear(hidden5_size, output_size)        

        def forward(self,x):
            out = F.leaky_relu(self.fc1(x))
            out = F.leaky_relu(self.fc2(out))
            out = F.leaky_relu(self.fc3(out))
            out = F.leaky_relu(self.fc4(out))
            out = F.leaky_relu(self.fc5(out))
            
            out = self.fc6(out)
            return out
        

def predict(pred_path):
    mod = torch.load("best.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    embedding_model = vggish()
    embedding_model.eval()

    try:
        example = vggish_input.wavfile_to_examples(pred_path)
        embeddings = embedding_model.forward(example)
        embeddings = embeddings.detach().numpy()
        embeddings/=255.0
        embeddings = embeddings.reshape(1,embeddings.shape[0]*embeddings.shape[1])
        embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
    except Exception as e:
        print("Error in extracting embedding from audio file: ", e)
        exit(1)


    out = mod(embeddings)
    pred = (torch.max(torch.exp(out), 1)[1]).data.cpu().numpy()
    classes = ("blues" , "classical" , "country" , "disco" , "hiphop" , "jazz" , "metal" , "pop" , "reggae" , "rock")
    print(classes[pred[0]])
    return classes[pred[0]]

# st.write('<span style="color: white; font-size: 48px;">ALGHAMI</span>', unsafe_allow_html=True)
st.write("<h1 style='color: purple;'>ALGHAMI Music Genre Classifier</h1>", unsafe_allow_html=True)

# st.title("Music Genre Classifier")

uploaded_file = st.file_uploader("Upload a WAV File", type=["wav"])

def display_wav_info(file):
    if file is not None:
        with st.spinner("Analyzing..."):
            try:
                wav = wave.open(file)
                wavv = AudioSegment.from_wav(file)
                wavv.export("file.wav", format="wav")
                cls = predict("file.wav")
                st.write('<span style="color: red; font-size: 24px;">The genre of the song is: ' + cls + ' 🎉' + '</span>', unsafe_allow_html=True)
                num_channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                frame_rate = wav.getframerate()
                num_frames = wav.getnframes()
                duration = num_frames / float(frame_rate)

                st.subheader("WAV File Information:")
                st.write(f"Number of Channels: {num_channels}")
                st.write(f"Sample Width (bytes): {sample_width}")
                st.write(f"Frame Rate (samples per second): {frame_rate}")
                st.write(f"Number of Frames: {num_frames}")
                st.write(f"Duration (seconds): {duration:.2f} seconds")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

display_wav_info(uploaded_file)