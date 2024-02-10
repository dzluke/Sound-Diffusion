import numpy as np
import math

def clear_dir(p):
    """
    Delete the contents of the directory at p
    """
    if not p.is_dir():
        return
    for f in p.iterdir():
        if f.is_file():
            f.unlink()
        else:
            clear_dir(f)


def format_time(seconds):
    """

    :param seconds:
    :return: a dictionary with the keys 'h', 'm', 's', that is the amount of hours, minutes, seconds equal to 'seconds'
    """
    hms = [seconds // 3600, (seconds // 60) % 60, seconds % 60]
    hms = [int(t) for t in hms]
    labels = ['h', 'm', 's']
    return {labels[i]: hms[i] for i in range(len(hms))}


def time_string(seconds):
    """
    Returns a string with the format "0h 0m 0s" that represents the amount of time provided
    :param seconds:
    :return: string
    """
    t = format_time(seconds)
    return "{}h {}m {}s".format(t['h'], t['m'], t['s'])


def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


def spectrum(audio, sr):
    """
    Return the magnitude spectrum and the frequency bins

    :returns amps, freqs: where the ith element of amps is the amplitude of the ith frequency in freqs
    """
    # TODO: Should this be the normalized magnitude spectrum?
    # calculate amplitude spectrum
    N = next_power_of_2(audio.size)
    fft = np.fft.rfft(audio, N)
    amplitudes = abs(fft)

    # get frequency bins
    frequencies = np.fft.rfftfreq(N, d=1. / sr)

    return amplitudes, frequencies

def centroid(audio, sr):
    """
    Compute the spectral centroid of the given audio.
    Spectral centroid is the weighted average of the frequencies, where each frequency is weighted by its amplitude

    the centroid is the sum across each frequency f and amplitude a: f * a / sum(a)

    :param audio: audio as a numpy array
    :param sr: the sampling rate of the audio
    """

    amps, freqs = spectrum(audio, sr)
    return sum(amps * freqs) / sum(amps)


def spread(audio, sr):
    """
    Compute the spectral spread of the given audio
    Spectral spread is the average each frequency component weighted by its amplitude and subtracted by the spectral centroid, t

    spread = sqrt(sum(amp(k) * (freq(k) - centroid)^2) / sum(amp))
    """
    amps, freqs = spectrum(audio, sr)
    cent = centroid(audio, sr)

    return math.sqrt(sum(amps * (freqs - cent)**2) / sum(amps))


def skewness(audio, sr):
    """
    Compute the spectral skewness

    Skewness is the sum of (freq - centroid)^3 * amps divided by (spread^3 times the sum of the amps)

    """
    amps, freqs = spectrum(audio, sr)
    cent = centroid(audio, sr)
    spr = spread(audio, sr)

    return sum(amps * (freqs - cent)**3) / (spr**3 * sum(amps))


def kurtosis(audio, sr):
    """
    Compute the spectral kurtosis

    Kurtosis is the sum of (freq - centroid)^4 * amp divided by (the spread^4 times the sum of the amps)

    """
    amps, freqs = spectrum(audio, sr)
    cent = centroid(audio, sr)
    spr = spread(audio, sr)

    return sum(amps * (freqs - cent)**4) / (spr**4 * sum(amps))


def moments(audio, sr):
    """
    Return the four statistical moments that make up the spectral shape: centroid, spread, skewness, kurtosis
    """
    amps, freqs = spectrum(audio, sr)

    cent = sum(amps * freqs) / sum(amps)
    spr = math.sqrt(sum(amps * (freqs - cent)**2) / sum(amps))
    skew = sum(amps * (freqs - cent)**3) / (spr**3 * sum(amps))
    kurt = sum(amps * (freqs - cent)**4) / (spr**4 * sum(amps))

    return cent, spr, skew, kurt



