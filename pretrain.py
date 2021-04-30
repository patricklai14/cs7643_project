import datasets

from transformers import DataCollatorForLanguageModeling
from transformers import LineByLineTextDataset
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import Trainer, TrainingArguments

import pdb

def main():

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForMaskedLM.from_pretrained('roberta-base')

    #train_data_file = "D:\\Work\\sandbox\\data\\train_citation.jsonl"
    train_data_file = "D:\\Work\\sandbox\\data\\train_scierc.jsonl"
    #train_data_file = "D:\\Work\\sandbox\\data\\train_chemprot.jsonl"
    train_data = datasets.load_dataset(
                    "json", data_files=train_data_file)["train"]

    def tokenization(batched_text):
        tokenized_batch = tokenizer(batched_text['text'], padding=True, truncation=True, return_special_tokens_mask=True)
        return tokenized_batch

    train_data = train_data.map(tokenization, batched=True, batch_size=len(train_data), remove_columns=["text", "label"])
    train_data.set_format('torch', columns=['input_ids'])

    #static masking. Possible TODO: use dynamic masking
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="./tapt_scierc",
        overwrite_output_dir=True,
        num_train_epochs=100,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=1e-4,
        #lr_scheduler_type="constant",        
        adam_epsilon=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.98,
        weight_decay=0.01,
        warmup_ratio=0.06,
        fp16=True,
        eval_accumulation_steps=20,
        save_steps=5000,
        save_total_limit=2,
        seed=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data
    )
    trainer.train()

    trainer.save_model("./tapt_scierc")

if __name__ == "__main__":
    main()