import argparse
import librosa
import numpy as np

# Function to find the sonic descriptors so i can import it in generate.py
def find_desceriptors(audio):
    # Calculate the amplitude
    # Compute the RMS value
    rms = np.sqrt(np.mean(audio**2))
    # Convert the RMS value to dB
    loudness_in_db = 20 * np.log10(rms)
    # Define the frame size and hop length for spectral centroid calculation
    FRAME_SIZE = 1024
    HOP_LENGTH = 512
    # Calculate the spectral centroid for each frame
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=44100, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    # Calculate the average spectral centroid
    avg_spectral_centroid = np.mean(spectral_centroid)
    return loudness_in_db, avg_spectral_centroid 

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--path",
#     type=str,
#     help="path to the audio to visualize",
#     required=True
# )

# args = parser.parse_args()
# AUDIO_PATH = args.path

# print(f"Processing {AUDIO_PATH}...")

# try:
#     # Load the audio file
#     y, sr = librosa.load(AUDIO_PATH, sr=None)
    
#     # Calculate the amplitude
#     # Compute the RMS value
#     rms = np.sqrt(np.mean(y**2))
#     # Convert the RMS value to dB
#     loudness_in_db = 20 * np.log10(rms)
#     print(f"Loudness of the audio file: {loudness_in_db:.2f} dB")
    
#     # Define the frame size and hop length for spectral centroid calculation
#     FRAME_SIZE = 1024
#     HOP_LENGTH = 512
#     # Calculate the spectral centroid for each frame
#     spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
#     # Calculate the average spectral centroid
#     avg_spectral_centroid = np.mean(spectral_centroid)
    
#     print(f"Average Spectral Centroid: {avg_spectral_centroid}")
# except Exception as e:
#     print(f"An error occurred: {e}")