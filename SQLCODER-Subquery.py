# === Logging ===
initialize_logger()
log("Starting continued fine-tuning of model on subquery dataset...")

# === HuggingFace Login ===
login_to_huggingface(token="YOUR_TOKEN")

# === Load PEFT Adapter Config ===
adapter_path = ""
peft_config = load_peft_config(adapter_path)

# === Load Base Model & Tokenizer ===
bnb_config = configure_8bit_quantization()
base_model = load_base_model(peft_config.base_model_name, quant_config=bnb_config)
tokenizer = load_tokenizer(peft_config.base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# === Prepare Model for Training ===
model = prepare_for_kbit_training(base_model)
model = load_lora_adapter(model, adapter_path)

# === Load and Process Dataset ===
data_path = "train.jsonl"
dataset = load_json_dataset(data_path)

def format_instruction(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

def tokenize(example):
    example["text"] = format_instruction(example)
    tokens = tokenizer(example["text"], padding="max_length", truncation=True, max_length=4096)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize)

# === Training Arguments ===
training_args = define_training_args(
    output_dir="",
    batch_size=2,
    grad_accum_steps=8,
    num_epochs=1,
    learning_rate=2e-5,
    logging_steps=1,
    save_strategy="epoch",
    bf16_supported=check_bf16_support()
)

# === Define Timer Callback ===
class TimerCallback:
    def on_log(...):
        calculate_eta_and_log()

# === Define Data Collator ===
data_collator = create_lm_data_collator(tokenizer, mlm=False)

# === Initialize Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    callbacks=[TimerCallback()]
)

# === Resume From Checkpoint If Available ===
checkpoint_dir = ""
last_checkpoint = get_checkpoint_if_exists(checkpoint_dir)

# === Start Training ===
log("Starting continued training now...")
trainer.train(resume_from_checkpoint=last_checkpoint)
log("Training complete.")

# === Save Model and Tokenizer ===
save_model(model, "")
save_tokenizer(tokenizer, "")

# === Push to HuggingFace Hub ===
log("Pushing updated model to HF Hub...")
push_model_to_hub(model, "")
push_tokenizer_to_hub(tokenizer, "")
log("All done.")
