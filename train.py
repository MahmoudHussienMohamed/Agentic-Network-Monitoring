import os
from dotenv import load_dotenv
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

load_dotenv()

MODEL_NAME = "meta-llama/Llama-3.2-3B"
HF_TOKEN = os.getenv("HF_TOKEN")

def train(agent_name, data_path, output_dir):
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN
    )
    
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # disable cache for training
    
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
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize, 
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset"
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=4,
        fp16=True,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,  # Keep only 2 checkpoints to save space
        optim="paged_adamw_8bit",
        report_to="none",
        warmup_steps=100,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,  # Save memory
        max_grad_norm=1.0,  # Gradient clipping
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