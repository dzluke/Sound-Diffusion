# Sound Diffusion: research journal

SD = Stable Diffusion

todo:
- [ ] update to SD 2.0 https://github.com/Stability-AI/stablediffusion

## Experiment 1: Sound-to-Image (November 2022)

A sound is input to SD and an image is created, essentially visualizing the sound.

SD works by encoding the text prompt as a matrix of dim (77, 768). I believe this is the featurized version of the prompt, but more research is necessary to figure out exactly what the encoding is (is it tokenization?). This matrix is what is passed to the model, and the model outputs an image.

<img width="898" alt="SD pipeline" src="https://user-images.githubusercontent.com/22928303/201449782-32b41f2e-4853-4ddf-be99-ff4e3283cbcd.png">


A first look showed that the data in the matrix are floats is in the range [?, ?]. 

### Test 1: Waveform representation

My first approach was to use a waveform (sample) representation of audio, and fill the feature matrix with the sample values of the audio. The first 768 samples of the audio become the first row of the feature matrix, the next 768 samples are the second row, and so on. At a sampling rate of 44.1 kHz, you can fit about 1.5 seconds on audio into the matrix. The sample values are NOT scaled, so they exist in the range [-1, 1].

examples:

| Audio | Example images |
| ----- | -------------- |
| archeos-bell | <img src="https://user-images.githubusercontent.com/22928303/201450266-3a5aeda1-d842-49e7-927f-26044fb286b5.png" width=300/> <img src="https://user-images.githubusercontent.com/22928303/201450277-9dd56cd7-d029-4e49-bb5d-51b635296058.png" width=300/> <img src="https://user-images.githubusercontent.com/22928303/201450428-25e85cdb-8ac4-4e7f-922d-bc7bda11a795.png" width=300/> |
| bass_clarinet   | <img src="https://user-images.githubusercontent.com/22928303/201450472-1ec6906c-54c0-4972-81b6-741e51beefce.png" width=300/> <img src="https://user-images.githubusercontent.com/22928303/201450487-9bc509f5-492f-4587-9948-3d614712b1f4.png" width=300/> <img src="https://user-images.githubusercontent.com/22928303/201450493-d6a55776-4313-44d8-83c7-db81d1991e32.png" width=300/> <img src="https://user-images.githubusercontent.com/22928303/201450509-89f0b9b0-348e-4c94-b729-4b914b952520.png" width=300/> |
| 1 Hz sine | <img src="https://user-images.githubusercontent.com/22928303/201450659-ffef3921-9dfb-45ab-815f-2230b1463bff.png" width=300/> |
| 10 Hz sine | <img src="https://user-images.githubusercontent.com/22928303/201450678-4c6dfa80-7331-480e-8cde-acfb44944183.png" width=300/> |
| 100 Hz sine | <img src="https://user-images.githubusercontent.com/22928303/201450684-83b29a47-001c-4013-b058-793e30c6adec.png" width=300/> <img src="https://user-images.githubusercontent.com/22928303/201450690-739b7c9a-bbae-424e-9d24-881cb2713c7c.png" width=300/> <img src="https://user-images.githubusercontent.com/22928303/201450696-c528514c-babb-48be-8976-f25bcd46f362.png" width=300/>  |


### Test 2: FFT representation

Due to the unstable nature of the waveform representation, I tried computing the STFT of the input sound and passing that as the input to SD. In order to make the dimensions of the STFT matrix fit the dimensions expected by SD, I had to use a number of frequency bins equal to 768. Only the real part of the STFT is kept. 

```
# stft makes an np array of size (1 + n_fft / 2, 1 + audio.size // hop_length)
n_fft = 1534
hop_length = y.size // 77 + 1
stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
stft = np.abs(stft)
```

Using the FFT representation provided images that were much more consistent (self-similar). For a given audio input, the images generated were much more similar to each other compared to the waveform representation. Interestingly, different audio inputs also gave similar image outputs.

examples:


| Audio | Example images |
| ----- | -------------- |
| archeos-bell | <img src="https://user-images.githubusercontent.com/22928303/201451019-d4498eb5-71db-42d3-a45c-5a54bae30fe9.png" width=300/> <img src="https://user-images.githubusercontent.com/22928303/201451029-60d056a0-2c44-4364-baf4-fcbaef63686f.png" width=300/> |
| bass_clarinet   | <img src="https://user-images.githubusercontent.com/22928303/201451136-66cac5d3-608b-459c-808b-43803946393e.png" width=300/> <img src="https://user-images.githubusercontent.com/22928303/201451137-8ceeb75d-61a9-4349-b8f6-d0dd076c04bd.png" width=300/>  |
| 1 Hz sine | <img src="https://user-images.githubusercontent.com/22928303/201451149-e734495a-79df-4ddc-b606-efc22ebd4a2c.png" width=300/> <img src="https://user-images.githubusercontent.com/22928303/201451152-11af743a-c133-45cd-a86d-dbd4344f10fb.png" width=300/> |
| 10 Hz sine | <img src="https://user-images.githubusercontent.com/22928303/201451195-26d5dc79-e5bb-40e9-a837-c0ff0d72e854.png" width=300/> <img src="https://user-images.githubusercontent.com/22928303/201451196-4316b0e8-cdd3-4de6-bfd1-f087980a4bc8.png" width=300/> |
| 100 Hz sine | <img src="https://user-images.githubusercontent.com/22928303/201451216-bcd210b5-3f8e-4408-aa19-25a35337c286.png" width=300/> <img src="https://user-images.githubusercontent.com/22928303/201451231-dd2ba7ef-cc36-42ef-93fb-03334c580efc.png" width=300/> |

## Next steps:
- Create a visualization that evolves over time as the audio changes. This can be done in two ways: 
1. at each time point, the image is generated by the previous image, conditioned on the current audio 
2. prompt interpolation: SD generates frames that are interpolations between two points in time

## Reference / Examples:

examples of animation with SD:
- https://www.instagram.com/p/Ci7Z8DZDMTG/
- Xander Steenbrugge "Voyage through Time": https://twitter.com/xsteenbrugge/status/1558508866463219712, https://www.youtube.com/watch?v=Bo3VZCjDhGI&feature=youtu.be&ab_channel=NeuralSynesthesia

OP-Z Real time sound to image with stable diffusion: https://modemworks.com/projects/op-z-stable-diffusion/
