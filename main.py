import dataset_builder
from transformers import ASTFeatureExtractor, ASTConfig, ASTForAudioClassification, TrainingArguments, Trainer
import torch
import numpy as np
import evaluate

def preprocess_audio(batch):
    wavs = [audio["array"] for audio in batch["input_values"]]
    inputs = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors="pt")
    return {model_input_name: inputs.get(model_input_name), "labels": list(batch["labels"])}

def generate_metrics(eval_pred):
    logits = eval_pred.predictions
    predictions = np.argmax(logits, axis=-1)
    metrics = accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
    metrics.update(precision.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
    metrics.update(recall.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
    metrics.update(f1.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
    return metrics

''' Loading the dataset '''
dataset = dataset_builder.generate_dataset()

pretrained_model = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model)
model_input_name = feature_extractor.model_input_names[0]
SAMPLING_RATE = feature_extractor.sampling_rate

''' Normalization of the audio features '''
feature_extractor.do_normalize = False 
mean = []
std = []

dataset = dataset.cast_column("audio", {"type": "audio", "sampling_rate": SAMPLING_RATE})
dataset = dataset.rename_column("audio", "input_values")
dataset.set_transform(preprocess_audio, output_all_columns=False)

for batch in dataset["train"]:
    input_tensor = torch.tensor(batch[model_input_name])
    mean.append(torch.mean(input_tensor))
    std.append(torch.std(input_tensor))

feature_extractor.mean = np.mean(mean)
feature_extractor.std = np.mean(std)
feature_extractor.do_normalize = True

''' Loading the model and training '''
config = ASTConfig.from_pretrained(pretrained_model)
config.num_labels = len(dataset["train"].features["labels"].names)
config.label2id = {label: i for i, label in enumerate(dataset["train"].features["labels"].names)}
config.id2label = {i: label for i, label in enumerate(dataset["train"].features["labels"].names)}

model = ASTForAudioClassification.from_pretrained(pretrained_model, config=config, ignore_mismatched_sizes=True)

training_args = TrainingArguments(
    output_dir="./results",
    logging_dir="./logs",
    report_to="tensorboard",
    learning_rate=5e-5,
    push_to_hub=False,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_strategy="steps",
    logging_steps=20
)

AVERAGE = "binary"

accuracy = evaluate.load("accuracy")
recall = evaluate.load("recall")
precision = evaluate.load("precision")
f1 = evaluate.load("f1")


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=generate_metrics
)

trainer.train()

print("Evaluating the model on the test set...")
metrics_test = trainer.evaluate(dataset["test"])
print("Test set metrics:", metrics_test)
