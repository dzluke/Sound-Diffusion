import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import librosa
import soundfile as sf
from pathlib import Path
from scipy.ndimage import zoom

# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


# SCALE = 28.0912
SAMPLING_RATE = 44100


def get_device():
    if(torch.cuda.is_available()):
        return 'cuda'
    elif(torch.backends.mps.is_available()):
        return 'mps'
    else:
        return 'cpu'


from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor


# load safety model
# safety_model_id = "CompVis/stable-diffusion-safety-checker"
# safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
# safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(get_device())
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


# def check_safety(x_image):
#     safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
#     x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
#     assert x_checked_image.shape[0] == len(has_nsfw_concept)
#     for i in range(len(has_nsfw_concept)):
#         if has_nsfw_concept[i]:
#             x_checked_image[i] = load_replacement(x_checked_image[i])
#     return x_checked_image, has_nsfw_concept


def main(args=None):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature",
        type=str,
        help="how to embed the audio in feature space; options: 'waveform', 'fft', 'melspectrogram', 'mfcc'"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        help="path to folder of input audio files"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
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
    # Add the text prompt and strength
    parser.add_argument(
        "--textprompt",
        type=str,
        help="Create a text prompt",
        nargs="?",
        default=""
    )
    parser.add_argument(
        "--textstrength",
        type=float,
        help="determine the strength of the text prompt",
        default=1
    )
    # if 'args' was not passed to this function, read from sys.argv, else read from the provided string 'args'
    if args is None:
        opt = parser.parse_args()
    else:
        opt = parser.parse_args(args)

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device(get_device())
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = Path(opt.outdir)

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    ###################
    # AUDIO DIFFUSION #
    ###################

    audio_files = []  # list of paths to audio inputs
    save_paths = []  # list of paths to save images
    data = []  # audio data embedded in feature space

    if opt.from_file:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            for p in f.read().splitlines():
                audio_files.append(Path(p))
    elif opt.input_folder:
        print(f"reading audio inputs from {opt.input_folder}")
        input_folder = Path(opt.input_folder)
        input_files = librosa.util.find_files(input_folder)
        for file in input_files:
            audio_files.append(Path(file))
    else:
        assert opt.prompt is not None
        audio_files.append(Path(opt.prompt))

    # load audio
    for audio_path in audio_files:
        audio_name = audio_path.stem
        save_path = outpath / audio_name
        os.makedirs(save_path, exist_ok=True)
        save_paths.append(save_path)
        y, _ = librosa.load(audio_path, sr=SAMPLING_RATE)

        if opt.feature == 'waveform':
            scale = 1
            y = librosa.util.fix_length(y, size=77 * 768)
            sf.write(save_path / f"{audio_name}.wav", y, samplerate=SAMPLING_RATE)
            c = np.resize(y, (1, 77, 768))
            c *= scale
        elif opt.feature == 'fft':
            scale = 1
            n_fft = 1534
            hop_length = y.size // 77 + 1
            stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
            stft = np.abs(stft)
            stft *= scale
            zoom_factor = (77 / stft.shape[0], 768 / stft.shape[1])
            c = np.array([zoom(stft, zoom_factor)])
        elif opt.feature == 'melspectrogram':
            scale = 1
            mel = librosa.feature.melspectrogram(y=y, sr=SAMPLING_RATE, n_mels=128, fmax=8000)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db + 80) / 80
            mel_db *= scale
            zoom_factor = (77 / mel_db.shape[0], 768 / mel_db.shape[1])
            c = np.array([zoom(mel_db, zoom_factor)])
        elif opt.feature == 'mfcc':
            scale = 1
            mfccs = librosa.feature.mfcc(y=y, sr=SAMPLING_RATE, n_mfcc=13)
            mfccs = mfccs - np.mean(mfccs, axis=1, keepdims=True)
            zoom_factor = (77 / mfccs.shape[0], 768 / mfccs.shape[1])
            c = np.array([zoom(mfccs, zoom_factor)])
            c *= scale
        else:
            raise NotImplementedError("Only 'waveform', 'fft', 'melspectrogram', and 'mfcc' are implemented features")
    
        c = torch.from_numpy(c).to(device)
        data.append(c)

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn(
            [opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device="cpu"
        ).to(torch.device(device))

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    if device.type == 'mps':
        precision_scope = nullcontext # have to use f32 on mps
    with torch.no_grad():
        with precision_scope(device.type):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for i in range(len(data)):
                        save_path = save_paths[i]
                        

                        # Add the text prompt to the data, scaling to --textstrength
                        textdata = model.get_learned_conditioning(opt.textprompt)
                        
                        maxtext = torch.max(textdata)
                        mintext = torch.min(textdata)
                        textrange = maxtext-mintext
                        
                        ### david's playground
                        maxtext = torch.max(textdata) / opt.textstrength
                        mintext = torch.min(textdata) / opt.textstrength
                        textrange = maxtext-mintext
                        ###
                        maxsound = torch.max(data[i])
                        minsound = torch.min(data[i]) 
                        soundrange = maxsound-minsound
                        
                        normdata = ((data[i]-minsound)*textrange/soundrange) + mintext
                        # print("playground")
                        # print(torch.max(normdata))
                        # print(torch.min(normdata))
                        # print(torch.max(textdata))
                        # print(torch.min(textdata))
                        c = normdata + textdata

                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])

                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image = x_samples_ddim

                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                        base_count = len([x for x in list(save_path.iterdir()) if x.suffix == '.png'])
                        if not opt.skip_save:
                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img = put_watermark(img, wm_encoder)
                                img.save(save_path / f"{opt.feature}{base_count:05}.png")
                                base_count += 1

                        if not opt.skip_grid:
                            all_samples.append(x_checked_image_torch)

                if not opt.skip_grid and opt.n_iter > 1:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(save_path, f'grid-{base_count:04}.png'))

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
