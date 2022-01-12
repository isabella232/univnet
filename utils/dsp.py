import math
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
import librosa
import soundfile as sf


class DSP:

    def __init__(self,
                 num_mels: int,
                 sample_rate: int,
                 hop_length: int,
                 win_length: int,
                 n_fft: int,
                 fmin: float,
                 fmax: float,
                 peak_norm: bool,
                 ) -> None:

        self.n_mels = num_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax

    def wav_to_mel(self, y: np.array, normalize=True) -> np.array:
        spec = librosa.stft(
            y=y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length)
        spec = np.abs(spec)
        mel = librosa.feature.melspectrogram(
            S=spec,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax)
        return mel

    def normalize(self, mel: np.array) -> np.array:
        mel = np.clip(mel, a_min=1.e-5, a_max=None)
        return np.log(mel)

    def denormalize(self, mel: np.array) -> np.array:
        return np.exp(mel)

    def trim_silence(self, wav: np.array) -> np.array:
        return librosa.effects.trim(wav, top_db=self.trim_silence_top_db, frame_length=2048, hop_length=512)[0]
