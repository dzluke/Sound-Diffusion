from subprocess import run
from pathlib import Path
from librosa.util import find_files
import time
from util import time_string

AUDIO_PATH = Path('./audio')
OUTPUT_PATH = Path('./video_outputs')
FPS_LIST = [10, 20]
STRENGTH_LIST = [0.2, 0.5, 0.7]

AUDIO_PATH.mkdir(exist_ok=True)
OUTPUT_PATH.mkdir(exist_ok=True)

tic = time.time()
for f in find_files(AUDIO_PATH):
    for fps in FPS_LIST:
        for strength in STRENGTH_LIST:
            video_name = "{}_{}fps_strength{}.mp4".format(Path(f).stem, str(fps), str(strength))
            if (OUTPUT_PATH / video_name).exists():
                continue  # don't make the video if it already exists
            command = ["python",
                       "audio2video.py",
                       "--path", str(f),
                       "--fps", str(fps),
                       "--strength", str(strength)]
            run(command)  # creates output.mp4
            save_path = OUTPUT_PATH / video_name
            Path('./output.mp4').rename(save_path)

print("Took", time_string(time.time() - tic))
print("Done.")
