import os
import json
import random
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline,
    set_seed
)
import torch


set_seed(42)

data_file = 'synthetic_code_switch_data.json'  

with open("data-4.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

def concat_dialogue(entry):
    return ' '.join(entry['dialogue'])


texts = [concat_dialogue(entry) for entry in data]

random.shuffle(texts)
split = int(0.8 * len(texts))
train_texts = texts[:split]
test_texts = texts[split:]

train_dataset = Dataset.from_dict({'text': train_texts})
test_dataset = Dataset.from_dict({'text': test_texts})

dataset_dict = DatasetDict({'train': train_dataset, 'test': test_dataset})

model_name = "xlm-roberta-base"  
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

model = AutoModelForMaskedLM.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="./code_switch_mlm",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,  
    gradient_accumulation_steps=4,  
    learning_rate=2e-5,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=True,
    evaluation_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

trainer.train()

fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
result = fill_mask("Hello, <mask> kya kr raha hai?")
print(result)

from transformers import EvalPrediction
import numpy as np

def compute_metrics(p: EvalPrediction):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
    perplexity = torch.exp(torch.tensor(loss))
    return {"perplexity": perplexity.item()}

training_args_eval = TrainingArguments(
    output_dir="./code_switch_mlm_eval",
    prediction_loss_only=True,
    report_to="none",
)

trainer_eval = Trainer(
    model=model,
    args=training_args_eval,
    data_collator=data_collator,
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

eval_results = trainer_eval.evaluate()
print(eval_results)