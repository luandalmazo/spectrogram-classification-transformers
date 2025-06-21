''' Augmentating from train/with-speech-disorder (124) and train/without-speech-disorder (1304) to balance the dataset '''

import os
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import librosa
import soundfile as sf

AUGMENTATIONS = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(p=0.5)
])

''' augment audios randomly until achieve the target number of files (1304) '''
def augment_audio_files(directory, target_number):
    files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    current_count = len(files)
    
    if current_count >= target_number:
        print(f"Directory '{directory}' already has enough files ({current_count}). No augmentation needed.")
        return
    
    print(f"Augmenting audio files in '{directory}' to reach {target_number} files...")
    
    while current_count < target_number:
        for file in files:
            if current_count >= target_number:
                break
            
            file_path = os.path.join(directory, file)
            audio, sample_rate = librosa.load(file_path, sr=None)
            
            augmented_audio = AUGMENTATIONS(samples=audio, sample_rate=sample_rate)
            augmented_file_path = os.path.join(directory, f"augmented_{current_count}.wav")
            
            sf.write(augmented_file_path, augmented_audio, sample_rate)
            current_count += 1
            
            print(f"Created: {augmented_file_path} (Total: {current_count})")
        


    print(f"Augmentation complete. Total files in '{directory}': {current_count}")

if __name__ == "__main__":
  
    WITHOUT_SPEECH_DISORDER_DIR = './split_dataset/train/with-speech-disorder'
    target_number = 1304 

    augment_audio_files(WITHOUT_SPEECH_DISORDER_DIR, target_number)
    
    print("Audio augmentation completed successfully.")