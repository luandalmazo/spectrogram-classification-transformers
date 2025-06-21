from datasets import load_dataset

def generate_dataset():
    dataset = load_dataset(
        "audiofolder",
        data_dir="split_dataset",
        split={
            "train": "train",
            "validation": "validation",
            "test": "test"
        }
    )
    return dataset
