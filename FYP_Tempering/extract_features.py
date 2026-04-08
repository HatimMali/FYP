import os
import numpy as np
import librosa
from tqdm import tqdm

# ---------------- CONFIG ----------------
DATASET_PATH = "dataset"
OUTPUT_FEATURES = "features.npy"
OUTPUT_LABELS = "labels.npy"

AUTHENTIC_LABEL = 0
TAMPERED_LABEL = 1
SUPPORTED_EXT = {".wav", ".flac"}

SEGMENT_DURATION = 1.0
SR = 22050
# ----------------------------------------


# ✅ FIX 1: Recursive file loading
def collect_files(dataset_dir):
    entries = []

    for label_name, label in [("authentic", AUTHENTIC_LABEL), ("tampered", TAMPERED_LABEL)]:
        folder = os.path.join(dataset_dir, label_name)

        for root, _, files in os.walk(folder):   # 🔥 recursive
            for f in files:
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXT:
                    entries.append((os.path.join(root, f), label))

    return entries


# ---------------- SEGMENTATION ----------------
def segment_audio(y, sr):
    seg_len = int(SEGMENT_DURATION * sr)
    segments = []

    for i in range(0, len(y) - seg_len + 1, seg_len):
        seg = y[i:i + seg_len]
        if len(seg) == seg_len:
            segments.append(seg)

    return segments


# ---------------- FEATURE EXTRACTION ----------------
def extract_segment_features(segment, sr):
    # MFCC
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # ✅ FIX 2: Proper delta MFCC
    delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta, axis=1)

    # Spectral features
    centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr))  # 🔥 added

    # Time-domain features
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=segment))
    rms = np.mean(librosa.feature.rms(y=segment))

    return np.concatenate([
        mfcc_mean,
        delta_mean,
        [centroid, bandwidth, rolloff, zcr, rms]
    ])


# ---------------- AGGREGATION ----------------
def aggregate(features):
    mean = np.mean(features, axis=0)
    var = np.var(features, axis=0)

    if len(features) > 1:
        maxdiff = np.max(np.abs(np.diff(features, axis=0)), axis=0)
    else:
        maxdiff = np.zeros(features.shape[1])

    return np.concatenate([mean, var, maxdiff])


# ---------------- MAIN ----------------
def main():
    entries = collect_files(DATASET_PATH)

    print("Total files found:", len(entries))  # 🔥 debug

    X, y = [], []

    for path, label in tqdm(entries):
        try:
            audio, sr = librosa.load(path, sr=SR)

            segments = segment_audio(audio, sr)
            if len(segments) == 0:
                continue

            seg_feats = np.array([
                extract_segment_features(s, sr) for s in segments
            ])

            file_feat = aggregate(seg_feats)

            X.append(file_feat)
            y.append(label)

        except Exception as e:   # 🔥 FIX 3: show errors
            print("\nError in file:", path)
            print(e)
            continue

    X = np.array(X)
    y = np.array(y)

    np.save(OUTPUT_FEATURES, X)
    np.save(OUTPUT_LABELS, y)

    print("\nSaved features:", X.shape)
    print("Saved labels:", y.shape)


if __name__ == "__main__":
    main()