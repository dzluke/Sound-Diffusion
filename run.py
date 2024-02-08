from subprocess import run
from pathlib import Path
from librosa.util import find_files
import time
from util import time_string

AUDIO_PATH = Path('./inputs/Singularity.wav')
OUTPUT_PATH = Path('./video_outputs')
FPS_LIST = [20]
STRENGTH_LIST = [x / 10 for x in range(11)]

# AUDIO_PATH.mkdir(exist_ok=True)
OUTPUT_PATH.mkdir(exist_ok=True)

tic = time.time()
# for f in find_files(AUDIO_PATH):
f = AUDIO_PATH
for fps in FPS_LIST:
    for strength in STRENGTH_LIST:
        video_name = "strength{}_{}fps.mp4".format(Path(f).stem, str(strength), str(fps))
        if (OUTPUT_PATH / video_name).exists():
            continue  # don't make the video if it already exists

        prompt_file = Path('./inputs') / "{}.txt".format(AUDIO_PATH.stem)

        command = ["python",
                   "musicvideogenerator.py",
                   "--path", str(f),
                   "--prompt_file", str(prompt_file),
                   "--fps", str(fps),
                   "--strength", str(strength)]
        run(command)  # creates output.mp4
        save_path = OUTPUT_PATH / video_name
        Path('./output.mp4').rename(save_path)

print("Took", time_string(time.time() - tic))
print("Done.")
