import os
import shutil

DATASET_DIR = './dataset/'
SPLIT_DATASET_DIR = './split_dataset/'
PERCENTAGE_TRAIN = 0.8
PERCENTAGE_TEST = 0.1

if not os.path.exists(SPLIT_DATASET_DIR):
    os.makedirs(SPLIT_DATASET_DIR)

def split_dataset():
    for subdir in ['with-speech-disorder', 'without-speech-disorder']:
        subdir_path = os.path.join(DATASET_DIR, subdir)
        files = os.listdir(subdir_path)
        total_files = len(files)
        
        train_count = int(total_files * PERCENTAGE_TRAIN)
        test_count = int(total_files * PERCENTAGE_TEST)
        
        train_files = files[:train_count]
        test_files = files[train_count:train_count + test_count]
        validation_files = files[train_count + test_count:]

        for split, split_files in zip(['train', 'test', 'validation'], [train_files, test_files, validation_files]):
            split_dir = os.path.join(SPLIT_DATASET_DIR, split, subdir)
            if not os.path.exists(split_dir):
                os.makedirs(split_dir)
            for file in split_files:
                shutil.move(os.path.join(subdir_path, file), os.path.join(split_dir, file))

if __name__ == "__main__":
    split_dataset()
    print("Dataset split completed successfully.")
    print(f"Train: {PERCENTAGE_TRAIN*100}%, Test: {PERCENTAGE_TEST*100}%, Validation: {(1 - PERCENTAGE_TRAIN - PERCENTAGE_TEST)*100}%")