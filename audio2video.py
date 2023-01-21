# from scripts.img2img import main
import argparse
from pathlib import Path
import audio2img
import librosa
import generate
import numpy as np
from subprocess import run

FEATURE = "waveform"
SAMPLING_RATE = 44100
FRAME_RATE = 1
ENCODING_DIMENSION = (77, 768)

AUDIO_PATH = "/Users/luke/CNMAT/Heliosonos/samples/chords.wav"
IMAGE_STORAGE_PATH = Path("./image_outputs")
OUTPUT_VIDEO_PATH = Path("./output.mp4")


IMAGE_STORAGE_PATH.mkdir(exist_ok=True)

# perform audio2img for the first time step
# args = ["--plms", "--feature", FEATURE, "--prompt", FIRST_TIME_STEP_AUDIO]
# audio2img.main(args.join(" "))
# todo get path from call to audio2img
# perform img2img, conditioned on the next step of audio
# for _ in []: # for each time step of audio
#     pass
    # get the last image
    # create a new image using img2img, with the prompt being the current audio


encodings = []
scale = 1
# stft makes an np array of size (1 + n_fft / 2, 1 + audio.size // hop_length)
y, _ = librosa.load(AUDIO_PATH, sr=SAMPLING_RATE)
audio_length = y.size // SAMPLING_RATE
num_frames = audio_length // FRAME_RATE
n_fft = (ENCODING_DIMENSION[1] - 1) * 2  # 768 - 1 * 2 = 1534
hop_length = y.size // (ENCODING_DIMENSION[0] * num_frames) + 1

stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
stft = np.abs(stft)  # TODO: is it ok to throw out the phase data?
stft *= scale
frames = []
for i in range(num_frames):
    frame = stft[:, i * ENCODING_DIMENSION[0]:(i + 1) * ENCODING_DIMENSION[0]]
    frames.append(frame)

frames = [np.array([f.T]) for f in frames]  # transpose and add a dimension to have shape (1,77,768)
frames = np.array(frames)  # shape (num_frames,1,77,768)

# interpolation between chords
# interpolation = np.linspace(frames[0], frames[3], num=4)

# generate images
# generate.main(frames, IMAGE_STORAGE_PATH)

# turn images into video
ffmpeg_command = ["ffmpeg",
                  "-y",  # automatically overwrite if output exists
                  "-framerate", str(FRAME_RATE),  # set framerate
                  "-i", str(IMAGE_STORAGE_PATH) + "/%05d.png",  # set image source
                  "-i", str(AUDIO_PATH),  # set audio path
                  "-vcodec", "libx264",
                  # "-acodec", "copy",
                  "-pix_fmt", "yuv420p",
                  str(OUTPUT_VIDEO_PATH)]
run(ffmpeg_command)
