import os
from dotenv import load_dotenv
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass
from typing import Any, Dict, List
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

load_dotenv()

MODEL_NAME = "meta-llama/Llama-3.2-3B"
HF_TOKEN = os.getenv("HF_TOKEN")

@dataclass
class DataCollatorForCompletionOnly:
    """Custom data collator that properly handles padding for causal LM."""
    tokenizer: PreTrainedTokenizerBase
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        
        max_length = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_labels = []
        attention_mask = []
        
        for ids, lbls in zip(input_ids, labels):
            padding_length = max_length - len(ids)
            
            padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * padding_length)
            
            padded_labels.append(lbls + [-100] * padding_length)
            
            attention_mask.append([1] * len(ids) + [0] * padding_length)
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

def train(agent_name, data_path, output_dir):
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN
    )
    
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    lora = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj",
            "o_proj", "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=data_path)

    def tokenize(x):
        text = x["prompt"] + "\n" + x["completion"]
        
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors=None,
        )
        
        return {
            "input_ids": tokenized["input_ids"],
            "labels": tokenized["input_ids"],  
        }

    tokenized_dataset = dataset.map(
        tokenize, 
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset"
    )
    
    # Use custom data collator
    data_collator = DataCollatorForCompletionOnly(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=4, # replaced to 10 for consultant
        fp16=True,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        optim="adamw_torch",
        report_to="none",
        warmup_steps=100,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        logging_first_step=True,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )

    print(f"Training agent: {agent_name}")
    print(f"Dataset size: {len(tokenized_dataset['train'])} examples")
    
    trainer.train()
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python train.py <agent_name> <data_path> <output_dir>")
        sys.exit(1)
    
    train(sys.argv[1], sys.argv[2], sys.argv[3])