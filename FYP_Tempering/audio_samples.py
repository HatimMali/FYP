"""
Select 2000 valid LibriSpeech audio files (>5 sec) and copy to a new folder.

Usage:
    python select_librispeech_samples.py
"""

import os
import random
import shutil
import librosa

# ---------------- CONFIG ----------------
SOURCE_DIR = "LibriSpeech/train-clean-100"
DEST_DIR = "authentic_audio"
NUM_SAMPLES = 2000
MIN_DURATION = 5.0  # seconds
SEED = 42
# ----------------------------------------

def collect_all_flac_files(source_dir):
    all_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".flac"):
                all_files.append(os.path.join(root, file))
    return all_files


def filter_by_duration(file_list, min_duration):
    valid_files = []
    
    print("\nFiltering files by duration...")
    
    for i, file in enumerate(file_list, 1):
        try:
            y, sr = librosa.load(file, sr=None)
            duration = len(y) / sr
            
            if duration >= min_duration:
                valid_files.append(file)
                
        except Exception as e:
            continue
        
        if i % 500 == 0:
            print(f"Checked {i}/{len(file_list)} files...")
    
    return valid_files


def copy_selected_files(selected_files, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    
    print("\nCopying selected files...")
    
    for i, file_path in enumerate(selected_files, 1):
        filename = os.path.basename(file_path)
        dest_path = os.path.join(dest_dir, filename)
        
        try:
            shutil.copy2(file_path, dest_path)
        except Exception as e:
            print(f"Error copying {filename}: {e}")
            continue
        
        if i % 100 == 0:
            print(f"Copied {i}/{len(selected_files)} files...")
    
    print("\nCopying completed!")


def main():
    random.seed(SEED)
    
    print("=" * 60)
    print("LibriSpeech Sample Selector")
    print("=" * 60)
    
    # Step 1: Collect files
    print("\nScanning dataset...")
    all_files = collect_all_flac_files(SOURCE_DIR)
    print(f"Total .flac files found: {len(all_files)}")
    
    # Step 2: Filter by duration
    valid_files = filter_by_duration(all_files, MIN_DURATION)
    print(f"Valid files (> {MIN_DURATION} sec): {len(valid_files)}")
    
    if len(valid_files) < NUM_SAMPLES:
        print("ERROR: Not enough valid files!")
        return
    
    # Step 3: Random selection
    selected_files = random.sample(valid_files, NUM_SAMPLES)
    print(f"\nSelected {NUM_SAMPLES} random files.")
    
    # Step 4: Copy files
    copy_selected_files(selected_files, DEST_DIR)
    
    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Files saved in: {DEST_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()