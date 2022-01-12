import random
import subprocess
import numpy as np
import librosa
from scipy.io.wavfile import read


def get_commit_hash():
    message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return message.strip().decode('utf-8')

def read_wav_np(path)

    wav, sr = librosa.load(path, sr=22050)
    wav = wav.astype(np.float32)

    return sr, wav
