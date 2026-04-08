import os
import random
import shutil
import numpy as np
import librosa
import soundfile as sf

from pydub import AudioSegment
import tempfile

# -------- CONFIG --------
INPUT_DIR = "authentic_audio"
OUTPUT_DIR = "dataset"
SEED = 42
# ------------------------

random.seed(SEED)
np.random.seed(SEED)

SUPPORTED_EXT = {".wav", ".flac"}

# ---------------- AUDIO IO ----------------

def load_audio(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    return y.astype(np.float32), sr

def save_audio(path, y, sr):
    y = np.clip(y, -1, 1)
    sf.write(path, y, sr)

# ---------------- TAMPERING ----------------

def tamper_speed(y, sr):
    return librosa.effects.time_stretch(y, rate=random.uniform(0.7, 1.3))

def tamper_pitch(y, sr):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=random.uniform(-4, 4))

def tamper_noise(y, sr):
    snr = random.uniform(10, 30)
    power = np.mean(y**2)
    noise_power = power / (10**(snr/10))
    noise = np.random.normal(0, np.sqrt(noise_power), y.shape)
    return y + noise

def tamper_compress(y, sr):
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "a.wav")
        mp3 = os.path.join(tmp, "b.mp3")
        out = os.path.join(tmp, "c.wav")

        sf.write(wav, y, sr)
        AudioSegment.from_wav(wav).export(mp3, format="mp3", bitrate="32k")
        AudioSegment.from_mp3(mp3).export(out, format="wav")

        y2, _ = librosa.load(out, sr=sr)
    return y2

def tamper_splice(y, sr):
    cut = int(random.uniform(0.8, 1.5) * sr)
    if len(y) < cut * 3:
        return y

    start = random.randint(0, len(y) - cut)
    donor = random.randint(0, len(y) - cut)

    return np.concatenate([y[:start], y[donor:donor+cut], y[start:]])

TECHS = [tamper_speed, tamper_pitch, tamper_noise, tamper_compress, tamper_splice]

# ---------------- MAIN ----------------

def collect_files():
    files = []
    for root, _, fs in os.walk(INPUT_DIR):
        for f in fs:
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXT:
                files.append(os.path.join(root, f))
    return files

def main():
    os.makedirs(f"{OUTPUT_DIR}/authentic", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/tampered", exist_ok=True)

    files = collect_files()

    for f in files:
        name = os.path.basename(f)

        # copy authentic
        shutil.copy(f, f"{OUTPUT_DIR}/authentic/{name}")

        y, sr = load_audio(f)

        # 🔥 multiple tampering
        y_t = y.copy()
        for tech in random.sample(TECHS, random.randint(1, 3)):
            try:
                y_t = tech(y_t, sr)
            except:
                continue

        save_audio(f"{OUTPUT_DIR}/tampered/{name}", y_t, sr)

    print("Dataset created!")

if __name__ == "__main__":
    main()