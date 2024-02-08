from pathlib import Path
import librosa
import generate
import numpy as np
from subprocess import run
import time
import util
import argparse
from PIL import Image
import shutil
from omegaconf import OmegaConf


FEATURE = "waveform"
INIT_PROMPT = None
SAMPLING_RATE = 44100
ENCODING_DIMENSION = (77, 768)

IMAGE_STORAGE_PATH = Path("./image_outputs")
OUTPUT_VIDEO_PATH = Path("./video_outputs")
OUTPUT_VIDEO_PATH.mkdir(exist_ok=True)


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
    return rms, avg_spectral_centroid 



parser = argparse.ArgumentParser()
parser.add_argument(
    "--feature",
    type=str,
    help="how to embed the audio in feature space; options: 'waveform', 'fft', 'melspectrogram'",
    default="melspectrogram"
)
parser.add_argument(
    "--input_folder",
    type=str,
    help="path to folder of input audio files"
)
parser.add_argument(
    "--outdir",
    type=str,
    nargs="?",
    help="dir to write results to",
    default="outputs/audio2img-samples"
)
parser.add_argument(
    "--skip_grid",
    action='store_true',
    default=True,
    help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
)
parser.add_argument(
    "--skip_save",
    action='store_true',
    help="do not save individual samples. For speed measurements.",
)
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)
parser.add_argument(
    "--plms",
    action='store_true',
    help="use plms sampling",
)
parser.add_argument(
    "--laion400m",
    action='store_true',
    help="uses the LAION400M model",
)
parser.add_argument(
    "--fixed_code",
    action='store_true',
    help="if enabled, uses the same starting code across samples ",
)
parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
    "--n_iter",
    type=int,
    default=1,
    help="sample this often",
)
parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=1,
    help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
    "--n_rows",
    type=int,
    default=0,
    help="rows in the grid (default: n_samples)",
)
parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--from-file",
    type=str,
    help="if specified, load prompts from this file",
)
parser.add_argument(
    "--config",
    type=str,
    default="configs/stable-diffusion/v1-inference.yaml",
    help="path to config which constructs model",
)
parser.add_argument(
    "--ckpt",
    type=str,
    default="models/ldm/stable-diffusion-v1/model.ckpt",
    help="path to checkpoint of model",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="the seed (for reproducible sampling)",
)
parser.add_argument(
    "--precision",
    type=str,
    help="evaluate at this precision",
    choices=["full", "autocast"],
    default="autocast"
)
parser.add_argument(
    "--path",
    type=str,
    help="path to the audio to visualize",
    required=True
)
parser.add_argument(
    "--fps",
    type=int,
    help="frames per second",
    required=True
)
parser.add_argument(
    "--strength",
    type=float,
    default=0.5,
    help="for img2img: strength for noising/unnoising. "
         "1.0 corresponds to full destruction of information in init image"
)
parser.add_argument(
    "--init-img",
    type=str,
    nargs="?",
    help="path to the input image"
)
parser.add_argument(
    "--textstrength",
    type=float,
    help="determine the strength of the text prompt",
    default=1.0
)
parser.add_argument(
    "--textprompt",
    type=str,
    help="Create a text prompt",
    nargs="?",
    default=""
)
parser.add_argument(
    "--textpromptend",
    type=str,
    help="Create a text prompt",
    nargs="?",
    default=""
)
parser.add_argument(
    "--prompt_file",
    type=str,
    help="path to the .txt file containing prompts",
)
args = parser.parse_args()
FRAME_RATE = args.fps
AUDIO_PATH = Path(args.path)
INIT_IMG = args.init_img

# Load prompts from file if specified
# TODO: this could be wrong, depending on how the textfile is written
if args.prompt_file is not None:
    with open(args.prompt_file, "r") as f:
        lines = [line.strip() for line in f.readlines()]
        prompts = [line.split(":")[-1] for line in lines]
        times = [float(line.split(":")[0]) * 60 + float(line.split(":")[1]) for line in lines]
        frametimes = [int(time * FRAME_RATE) for time in times]
num_prompts = len(prompts)


tic = time.time()
IMAGE_STORAGE_PATH.mkdir(exist_ok=True)
util.clear_dir(IMAGE_STORAGE_PATH)

encodings = []
scale = 1
# stft makes an np array of size (1 + n_fft / 2, 1 + audio.size // hop_length)
y, _ = librosa.load(AUDIO_PATH, sr=SAMPLING_RATE)
splitfiles = []
frames = []
audio_length = y.size // SAMPLING_RATE
num_frames = audio_length * FRAME_RATE
for i in range(num_frames):
    splitfiles.append(y[i * SAMPLING_RATE // FRAME_RATE: (i + 1) * SAMPLING_RATE // FRAME_RATE])

for y in splitfiles:
    if args.feature == 'waveform':
        scale = 1
        y = librosa.util.fix_length(y, size=77 * 768)
        # convert audio to feature space
        c = np.resize(y, (1, 77, 768))
        c *= scale
    elif args.feature == 'fft':
        scale = 1
        # stft makes an np array of size (1 + n_fft / 2, 1 + audio.size // hop_length)
        n_fft = 1534
        hop_length = y.size // 77 + 1
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        stft = np.abs(stft)  # TODO: is it ok to throw out the phase data?
        stft *= scale
        c = np.array([stft.T])  # transpose and add a dimension to have shape (1,77,768)
    elif args.feature == 'melspectrogram':
        scale = 1
        mel = librosa.feature.melspectrogram(y=y, sr=SAMPLING_RATE, n_mels=128, fmax=8000)  # adjust parameters accordingly
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db + 80) / 80  # normalize to [0,1]; -80dB is a common threshold for silence in audio
        mel_db *= scale
        # Resize mel_db to fit the desired shape:
        c = np.resize(mel_db, (1, 77, 768))
    else:
        raise NotImplementedError("Only 'waveform', 'fft', and 'melspectrogram' are implemented features")
    frames.append(c)

# Check to see if endtext is empty, if so, replace with the same text as the prompt
if args.textpromptend == "":
    args.textpromptend = args.textprompt

rmsarray = []

print(">>> Loading model")
# load model
config = OmegaConf.load(f"{args.config}")
model = generate.load_model_from_config(config, f"{args.ckpt}")

print(">>> Generating {} images".format(num_frames))
print(">>> Using img2img")

# generate first frame
c = model.get_learned_conditioning(prompts[0])
generate.text2img(np.array([c.cpu()]), IMAGE_STORAGE_PATH, args)

curr_prompt = 0
next_prompt = 1
# for the rest of the images, each one is the previous image conditioned on the current prompt
for i in range(1, num_frames):
    prompt = frames[i]
    init_img_path = IMAGE_STORAGE_PATH / f"{(i - 1):05}.png"  # use the previous image
    # Add the MODEL_CKPT as input for the img2img function.
    args.seed = args.seed + 1

    print("current iteration: " + str(i))
    rms, spectral = find_desceriptors(splitfiles[i])

    if i >= frametimes[next_prompt]:
        curr_prompt = next_prompt
        next_prompt += 1
        if next_prompt >= num_prompts:
            next_prompt = curr_prompt

    args.textprompt = prompts[curr_prompt]
    args.textpromptend = prompts[next_prompt]
    curr_num_frames = frametimes[next_prompt] - frametimes[curr_prompt]
    if curr_num_frames == 0:
        print("ERROR: curr_num_frames == 0; skipping iteration")
        continue
    rmsarray.append(rms)
    generate.img2img(model, np.array([prompt]), init_img_path, IMAGE_STORAGE_PATH, i - frametimes[curr_prompt], curr_num_frames, rms, args)

    if i % (10 * FRAME_RATE) == 0:
        # every 10 seconds, create an in progress video
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



print(rmsarray)
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
print(">>> Avg time per frame: ", util.time_string((time.time() - tic) / num_frames))

# store an extra copy just in case
VIDEO_OUTPUT_FOLDER = Path("./video_outputs")
num_videos = len(list(VIDEO_OUTPUT_FOLDER.iterdir()))
filename = "output{}.mp4".format(num_videos)
shutil.copy(OUTPUT_VIDEO_PATH, VIDEO_OUTPUT_FOLDER / filename)

print("Done.")
