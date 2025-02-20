from datasets import Dataset, Audio, ClassLabel, Features
import pandas

def generate_dataset():
    ''' Defining class labels '''
    class_labels = ClassLabel(names = ["speech-disorder", "no-speech-disorder"])

    ''' Defining features '''
    features = Features({
        "audio": Audio(),
        "labels": class_labels
    })

    ''' Load csv '''
    df = pandas.read_csv("dataset_info.csv")

    ''' Create dataset '''
    dataset = Dataset.from_pandas(df, features = features)

    return dataset
