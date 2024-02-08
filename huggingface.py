"""
Reference: https://huggingface.co/docs/diffusers/using-diffusers/img2img
"""

import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
from pathlib import Path
from subprocess import run



IMAGE_STORAGE_PATH = Path("./image_outputs")
STRENGTH = 0.5
FRAME_RATE = 20
num_frames = 20


pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

# optimization
torch.backends.cuda.matmul.allow_tf32 = True

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-sdxl-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

for i in range(num_frames):
    # pass prompt and image to pipeline
    image = pipeline(prompt, image=init_image, strength=STRENGTH).images[0]
    filename = IMAGE_STORAGE_PATH / f'{i:05}.png'
    image.save(filename)
    init_image = image

# turn images into video
ffmpeg_command = ["ffmpeg",
                  "-y",  # automatically overwrite if output exists
                  "-framerate", str(FRAME_RATE),  # set framerate
                  "-i", str(IMAGE_STORAGE_PATH) + "/%05d.png",  # set image source
                  "-vcodec", "libx264",
                  "-pix_fmt", "yuv420p",
                  "output.mp4"]
run(ffmpeg_command)
