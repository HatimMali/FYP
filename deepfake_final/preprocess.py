import numpy as np
import librosa
import scipy.fftpack as fft
from scipy.signal import lfilter

# SAME CONFIG
SAMPLE_RATE = 16000
FRAME_LENGTH = 400
FRAME_SHIFT = 160
FFT_SIZE = 512
NUM_FILTERS = 20
NUM_LFCC = 20
FIXED_FRAMES = 400


def pre_emphasis(signal, coeff=0.97):
    return lfilter([1, -coeff], [1], signal)


def frame_signal(signal):
    signal_length = len(signal)
    num_frames = int(np.ceil((signal_length - FRAME_LENGTH) / FRAME_SHIFT)) + 1

    pad_length = (num_frames - 1) * FRAME_SHIFT + FRAME_LENGTH
    pad_signal = np.append(signal, np.zeros(pad_length - signal_length))

    frames = np.zeros((num_frames, FRAME_LENGTH))
    for i in range(num_frames):
        start = i * FRAME_SHIFT
        frames[i] = pad_signal[start:start + FRAME_LENGTH]

    return frames


def compute_lfcc(signal):
    signal = pre_emphasis(signal)
    frames = frame_signal(signal)

    window = np.hamming(FRAME_LENGTH)
    frames *= window

    spectrum = np.abs(np.fft.rfft(frames, FFT_SIZE)) ** 2

    # Linear filter bank
    filters = np.linspace(0, SAMPLE_RATE // 2, NUM_FILTERS + 2)
    bins = np.floor((FFT_SIZE + 1) * filters / SAMPLE_RATE).astype(int)

    fbanks = np.zeros((NUM_FILTERS, FFT_SIZE // 2 + 1))

    for i in range(1, NUM_FILTERS + 1):
        left, center, right = bins[i-1], bins[i], bins[i+1]

        for j in range(left, center):
            fbanks[i-1, j] = (j - left) / (center - left)
        for j in range(center, right):
            fbanks[i-1, j] = (right - j) / (right - center)

    features = np.dot(spectrum, fbanks.T)
    features = np.where(features == 0, np.finfo(float).eps, features)

    log_features = np.log(features)
    lfcc = fft.dct(log_features, type=2, axis=1, norm='ortho')[:, :NUM_LFCC]

    return lfcc.T  # (20, frames)


def pad_or_truncate(features):
    if features.shape[1] < FIXED_FRAMES:
        pad_width = FIXED_FRAMES - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)))
    else:
        features = features[:, :FIXED_FRAMES]

    return features


def extract_features(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    lfcc = compute_lfcc(audio)
    lfcc = pad_or_truncate(lfcc)

    return lfcc.astype(np.float32)  # (20, 400)