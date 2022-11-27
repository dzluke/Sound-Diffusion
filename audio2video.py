# from scripts.img2img import main
import argparse
from pathlib import Path
import audio2img

FEATURE = "waveform"
FIRST_TIME_STEP_AUDIO = Path("./temp.wav")

# perform audio2img for the first time step
args = ["--plms", "--feature", FEATURE, "--prompt", FIRST_TIME_STEP_AUDIO]
audio2img.main(args.join(" "))
# todo get path from call to audio2img
# perform img2img, conditioned on the next step of audio
for _ in []: # for each time step of audio
    # get the last image
    # create a new image using img2img, with the prompt being the current audio
