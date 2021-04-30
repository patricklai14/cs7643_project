import datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DataCollatorForLanguageModeling
from transformers import LineByLineTextDataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments

import numpy as np
import torch

import argparse
import pdb

#guides used:
#https://jesusleal.io/2020/10/20/RoBERTA-Text-Classification/
#https://huggingface.co/transformers/training.html

# define accuracy metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    parser = argparse.ArgumentParser(description="Run feature selection")
    parser.add_argument("--train_dataset", type=str, required=True,
                        help="dataset file to train classifier")
    parser.add_argument("--val_dataset", type=str, required=True,
                        help="dataset file for validation data")
    parser.add_argument("--test_dataset", type=str, required=True,
                        help="dataset file to evaluate classifier")
    parser.add_argument("--model", type=str, required=True,
                        help="location of pretrained model or \"roberta-base\" for base roberta")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="directory to save model")

    args = parser.parse_args()
    args_dict = vars(args)

    train_data = datasets.load_dataset(
                    "json", data_files=args_dict["train_dataset"])["train"]
    
    #not a typo: load_dataset loads this as training data
    val_data = datasets.load_dataset(
                    "json", data_files=args_dict["val_dataset"])["train"] 
    test_data = datasets.load_dataset(
                    "json", data_files=args_dict["test_dataset"])["train"] 

    #assign an integer key for each label
    label_keys = {}
    num_labels = 0
    for label in train_data['label']:
        if label not in label_keys:
            label_keys[label] = num_labels
            num_labels += 1

    for label in test_data['label']:
        if label not in label_keys:
            label_keys[label] = num_labels
            num_labels += 1

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def tokenization(batched_text):
        tokenized_batch = tokenizer(batched_text['text'], padding=True, truncation=True)
        tokenized_batch["label"] = [label_keys[label] for label in batched_text["label"]]
        return tokenized_batch

    train_data = train_data.map(tokenization, batched=True, batch_size=len(train_data))
    val_data = val_data.map(tokenization, batched=True, batch_size=len(val_data))
    test_data = test_data.map(tokenization, batched=True, batch_size=len(test_data))

    train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    val_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    f1_scores = []
    for seed in range(1, 4):
        model = RobertaForSequenceClassification.from_pretrained(
                args_dict["model"], hidden_dropout_prob=0.1, num_labels=num_labels)

        training_args = TrainingArguments(
            output_dir=args_dict["output_dir"],
            overwrite_output_dir=True,
            num_train_epochs=10,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=2e-5,
            lr_scheduler_type="constant",
            fp16=True,
            eval_accumulation_steps=20,
            save_strategy="no",
            save_steps=5000,
            save_total_limit=1,
            load_best_model_at_end=False,
            metric_for_best_model="f1",
            evaluation_strategy="epoch",
            seed=seed
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_data,
            eval_dataset=val_data
        )

        trainer.train()

        metrics = trainer.evaluate(test_data)
        print("Metrics for seed {}: {}".format(seed, metrics))

        f1_scores.append(metrics["eval_f1"])

    print("Avg f1 score: {}, stdev: {}".format(np.mean(f1_scores), np.std(f1_scores)))


if __name__ == "__main__":
    main()