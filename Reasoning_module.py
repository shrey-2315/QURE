function STaR_Reasoning_Workflow():
    # Step 1: Login to Hugging Face
    login_to_huggingface(HF_TOKEN)

    # Step 2: Load Model & Tokenizer with 8-bit Quantization
    model = load_model(MODEL_NAME, quantized=True)
    tokenizer = load_tokenizer(MODEL_NAME)
    apply_padding_token_if_missing(tokenizer)
    model = apply_peft_lora(model, LORA_CONFIG)
    print_trainable_params(model)
    clear_gpu_memory()

    # Step 3: Load Training and Test Data
    train_dataset, test_dataset = load_data()

    # Step 4: Phase 1 - Generate and Evaluate
    correct, failed = star_phase_1(train_dataset, model, tokenizer)
    clear_gpu_memory()

    # Step 5: Phase 2 - Regenerate Failed Examples
    regenerated = star_phase_2(model, tokenizer, failed)
    clear_gpu_memory()

    # Step 6: Phase 3 - Fine-tune on Correct + Regenerated Data
    fine_tune(model, tokenizer, correct + regenerated)
    clear_gpu_memory()

    # Step 7: Final Evaluation on Test Set
    evaluate(model, tokenizer, test_dataset)
    clear_gpu_memory()

    # Step 8: Upload Fine-Tuned Model to Hugging Face
    upload_model_to_hf(model)

    print("STaR pipeline completed.")

