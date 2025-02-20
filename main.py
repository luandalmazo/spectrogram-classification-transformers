import dataset_builder
from transformers import ASTFeatureExtractor, ASTConfig, ASTForAudioClassification, TrainingArguments, Trainer
import torch
import numpy as np
import evaluate

''' Generating dataset '''
dataset = dataset_builder.generate_dataset()

pretrained_model = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model)

model_input_name = feature_extractor.model_input_names[0]
SAMPLING_RATE = feature_extractor.sampling_rate

''' Calculating mean and std '''
feature_extractor.do_normalize = False 
mean = []
std = []

def preprocess_audio(batch):
    wavs = [audio["array"] for audio in batch["input_values"]]

    inputs = feature_extractor(wavs, sampling_rate = SAMPLING_RATE, return_tensors = "pt")

    output_batch = {model_input_name: inputs.get(model_input_name), "labels": list(batch["labels"])}

    return output_batch

dataset = dataset.rename_column("audio", "input_values")
dataset.set_transform(preprocess_audio, output_all_columns = False)

for batch in dataset:
    input_values = batch[model_input_name]
    input_tensor = torch.tensor(input_values)
    mean.append(torch.mean(input_tensor))
    std.append(torch.std(input_tensor))

feature_extractor.mean = np.mean(mean)
feature_extractor.std = np.mean(std)

''' Normalize '''
feature_extractor.do_normalize = True

''' Setting model '''
config = ASTConfig.from_pretrained(pretrained_model)
config.num_labels = len(dataset.features["labels"].names)
config.label2id = {label: i for i, label in enumerate(dataset.features["labels"].names)}
config.id2label = {i: label for i, label in enumerate(dataset.features["labels"].names)}

dataset = dataset.train_test_split(test_size = 0.2, shuffle=True)

model = ASTForAudioClassification.from_pretrained(pretrained_model, config = config, ignore_mismatched_sizes=True)
model.init_weights()

training_args = TrainingArguments(
    output_dir="./results",
    logging_dir="./logs",
    report_to="tensorboard",
    learning_rate=5e-5,
    push_to_hub=False,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    eval_steps=1,
    save_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_strategy="steps",
    logging_steps=20)


accuracy = evaluate.load("accuracy")
recall = evaluate.load("recall")
precision = evaluate.load("precision")
f1 = evaluate.load("f1")

AVERAGE = "binary"

def generate_metrics(eval_pred):
    logits = eval_pred.predictions
    predictions = np.argmax(logits, axis = -1)
    metrics = accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
    metrics.update(precision.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
    metrics.update(recall.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
    metrics.update(f1.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
    return metrics

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    compute_metrics = generate_metrics)

trainer.train()