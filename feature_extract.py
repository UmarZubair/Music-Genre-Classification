import numpy as np
from typing import Optional
from librosa.filters import mel
from librosa.core import load as lb_load, stft

def extract_mel_band_energies(audio_file: np.ndarray,
                              sr: Optional[int] = 44100,
                              n_fft: Optional[int] = 1024,
                              hop_length: Optional[int] = 512,
                              n_mels: Optional[int] = 40) \
                            -> np.ndarray:
    """Extracts and returns the mel-band energies from the `audio_file` audio file.

    :param audio_file: Path of the audio file.
    :type audio_file: str
    :param sr: Sampling frequency of audio file, defaults to 44100.
    :type sr: Optional[int]
    :param n_fft: STFT window length (in samples), defaults to 1024.
    :type n_fft: Optional[int]
    :param hop_length: Hop length (in samples), defaults to 512.
    :type hop_length: Optional[int]
    :param n_mels: Number of MEL frequencies/filters to be used, defaults to 40.
    :type n_mels: Optional[int]
    :return: Mel-band energies of the `audio_file` audio file.
    :rtype: numpy.ndarray
    """
    
    spec = stft(
        y=audio_file,
        n_fft=n_fft,
        hop_length=hop_length)

    mel_filters = mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

    return np.dot(mel_filters, np.abs(spec) ** 2)