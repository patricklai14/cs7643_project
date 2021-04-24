import datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DataCollatorForLanguageModeling
from transformers import LineByLineTextDataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments

import torch

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
    train_data = datasets.load_dataset(
                    "json", data_files="D:\\Work\\sandbox\\data\\train_citation.jsonl")["train"]
    
    #not a typo: load_dataset loads this as training data
    test_data = datasets.load_dataset(
                    "json", data_files="D:\\Work\\sandbox\\data\\test_citation.jsonl")["train"] 

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
    model = RobertaForSequenceClassification.from_pretrained(".\\roberta_tapt_citation", hidden_dropout_prob=0.1, num_labels=num_labels)
    #model = RobertaForSequenceClassification.from_pretrained('roberta-base', hidden_dropout_prob=0.1, num_labels=num_labels)

    def tokenization(batched_text):
        tokenized_batch = tokenizer(batched_text['text'], padding=True, truncation=True)
        tokenized_batch["label"] = [label_keys[label] for label in batched_text["label"]]
        return tokenized_batch

    train_data = train_data.map(tokenization, batched=True, batch_size=len(train_data))
    test_data = test_data.map(tokenization, batched=True, batch_size=len(test_data))

    train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    training_args = TrainingArguments(
        output_dir="./roberta_tapt_citation",
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        fp16=True,
        eval_accumulation_steps=20,
        save_steps=5000,
        save_total_limit=2,
        seed=3
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=test_data
    )

    trainer.train()
    print(trainer.evaluate())
    pdb.set_trace()

if __name__ == "__main__":
    main()