from pathlib import Path
import librosa
import generate
import numpy as np
from subprocess import run
import time
import util
import argparse
from PIL import Image


FEATURE = "waveform"
IMG2IMG = True  # use img2img to condition each generation on previous image
INIT_PROMPT = None
SAMPLING_RATE = 44100
ENCODING_DIMENSION = (77, 768)

IMAGE_STORAGE_PATH = Path("./image_outputs")
OUTPUT_VIDEO_PATH = Path("./output.mp4")

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--path",
    type=str,
    help="path to the audio to visualize",
    required=True
)
arg_parser.add_argument(
    "--fps",
    type=int,
    help="frames per second",
    required=True
)
arg_parser.add_argument(
    "--strength",
    type=float,
    default=0.5,
    help="for img2img: strength for noising/unnoising. "
         "1.0 corresponds to full destruction of information in init image"
)
arg_parser.add_argument(
    "--init-img",
    type=str,
    nargs="?",
    help="path to the input image"
)
arg_parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )

args = arg_parser.parse_args()
FRAME_RATE = args.fps
AUDIO_PATH = args.path
INIT_IMG = args.init_img
MODEL_CKPT = args.ckpt

tic = time.time()
IMAGE_STORAGE_PATH.mkdir(exist_ok=True)
util.clear_dir(IMAGE_STORAGE_PATH)

encodings = []
scale = 1
# stft makes an np array of size (1 + n_fft / 2, 1 + audio.size // hop_length)
y, _ = librosa.load(AUDIO_PATH, sr=SAMPLING_RATE)
audio_length = y.size // SAMPLING_RATE
num_frames = audio_length * FRAME_RATE
n_fft = (ENCODING_DIMENSION[1] - 1) * 2  # 768 - 1 * 2 = 1534
hop_length = y.size // (ENCODING_DIMENSION[0] * num_frames) - 1

stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
stft = np.abs(stft)  # TODO: is it ok to throw out the phase data?
stft *= scale
frames = []
for i in range(num_frames):
    frame = stft[:, i * ENCODING_DIMENSION[0] : (i + 1) * ENCODING_DIMENSION[0]]
    frames.append(frame)

frames = [np.array([f.T]) for f in frames]  # transpose and add a dimension to have shape (1,77,768)
frames = np.array(frames)  # shape (num_frames,1,77,768)

# interpolation between chords
# interpolation = np.linspace(frames[0], frames[3], num=4)

# generate images
print(">>> Generating {} images".format(num_frames))
if IMG2IMG:
    print(">>> Using img2img")
    if INIT_IMG is not None:
        # first frame is provided image
        img = Image.open(INIT_IMG)
        img = img.resize((512, 512))
        save_path = IMAGE_STORAGE_PATH / f"{0:05}.png"
        img.save(save_path)
    elif INIT_PROMPT is not None:
        # generate the first frame using text2img with text as prompt
        pass
    else:
        # generate the first frame using text2img with audio as prompt
        generate.text2img(np.array([frames[0]]), IMAGE_STORAGE_PATH, MODEL_CKPT)

    # for the rest of the images, each one is the previous image conditioned on the current prompt
    for i in range(1, num_frames):
        prompt = frames[i]
        init_img_path = IMAGE_STORAGE_PATH / f"{(i - 1):05}.png"  # use the previous image
        generate.img2img(np.array([prompt]), init_img_path, IMAGE_STORAGE_PATH, args.strength, MODEL_CKPT)
else:
    print(">>> Using text2img")
    # generate only using text2img
    generate.text2img(frames, IMAGE_STORAGE_PATH)

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

print(">>> Generated {} images".format(num_frames))
print(">>> Took", util.time_string(time.time() - tic))
print("Done.")
