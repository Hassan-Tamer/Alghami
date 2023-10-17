import streamlit as st
import wave

# Define the Streamlit app
st.title("WAV File Info")

# Create a file uploader
uploaded_file = st.file_uploader("Upload a WAV File", type=["wav"])

# Function to display information about the WAV file
def display_wav_info(file):
    if file is not None:
        with st.spinner("Analyzing..."):
            try:
                wav = wave.open(file)
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

# Display the WAV file information
display_wav_info(uploaded_file)
