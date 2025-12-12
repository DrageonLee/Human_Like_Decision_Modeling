import argparse
import os
import pandas as pd
import torch

from datasets import Dataset
from unsloth import FastLanguageModel
from trl import DPOTrainer
from transformers import TrainingArguments

# Import key-only prompt creator from utils
from utils import create_keyonly_prompt


def _normalize_choice(x):
    """
    Normalizes the judge's decision to 'A' or 'B'. Returns None if invalid.
    """
    if pd.isna(x):
        return None
    x = str(x).strip().upper()
    return x if x in ["A", "B"] else None


def prepare_dpo_dataset_keyonly(df: pd.DataFrame) -> list:
    """
    Constructs preference pairs for DPO training.
    
    CRITICAL: This function implements the 'Indirect Reasoning' approach (M5).
    - It uses the Judge's evaluation of the full reasoning traces (A vs B).
    - HOWEVER, the training data ('chosen'/'rejected') contains ONLY the final Key token.
    - This ensures the model learns to output the preferred Key without seeing reasoning at inference.

    Args:
        df: DataFrame with columns [Quiz, PredictionA, PredictionB, Judge]
    
    Returns:
        List of dicts: {'prompt': ..., 'chosen': 'K', 'rejected': 'J'}
    """
    required = ["Quiz", "PredictionA", "PredictionB", "Judge"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    dpo_data = []
    skipped_count = 0
    
    print(f"Processing {len(df)} rows for DPO...")

    for _, row in df.iterrows():
        judge = _normalize_choice(row["Judge"])
        if judge is None:
            skipped_count += 1
            continue

        quiz = str(row["Quiz"]).strip()
        if not quiz:
            continue

        # Extract just the key character (e.g., 'A')
        pred_a = str(row["PredictionA"]).strip().upper()
        pred_b = str(row["PredictionB"]).strip().upper()

        # Extract first letter only to ensure key-only format
        key_a = pred_a[0] if pred_a else ""
        key_b = pred_b[0] if pred_b else ""

        # Validate keys (must be single uppercase letters)
        if not (len(key_a) == 1 and key_a.isalpha() and len(key_b) == 1 and key_b.isalpha()):
            skipped_count += 1
            continue
            
        if key_a == key_b:
            # If both predictions resulted in the same key, DPO cannot learn a preference
            skipped_count += 1
            continue

        # Create the Key-Only prompt (Same format as Centaur/Evaluation)
        prompt = create_keyonly_prompt(quiz)

        if judge == "A":
            chosen, rejected = key_a, key_b
        else:
            chosen, rejected = key_b, key_a

        dpo_data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    print(f"Skipped {skipped_count} invalid or identical pairs.")
    return dpo_data


def main(args):
    # --- 1. Load Data ---
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    print(f"Loading judged data from: {args.data_path}")
    df = pd.read_csv(args.data_path)

    dpo_list = prepare_dpo_dataset_keyonly(df)
    if len(dpo_list) == 0:
        raise ValueError("No valid DPO pairs found. Check your CSV data.")

    dpo_dataset = Dataset.from_list(dpo_list)
    print(f"Successfully constructed {len(dpo_dataset)} key-only preference pairs.")

    # --- 2. Load Base Model ---
    print(f"Loading base model: {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    if "gemma" in args.base_model.lower():
        tokenizer.padding_side = "right"

    # --- 3. Apply LoRA Adapters ---
    print("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=args.seed,
    )

    # --- 4. Training Arguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )

    # --- 5. Initialize DPO Trainer ---
    trainer = DPOTrainer(
        model=model,
        ref_model=None, # Unsloth handles this efficiently
        args=training_args,
        beta=args.beta,
        train_dataset=dpo_dataset,
        tokenizer=tokenizer,
        # Key-only prompts are short, responses are 1 token.
        max_prompt_length=min(1024, args.max_seq_length),
        max_length=min(1152, args.max_seq_length),
    )

    # --- 6. Start Training ---
    print("Starting Key-Only DPO training...")
    trainer.train()

    # --- 7. Save Adapter ---
    print(f"Saving DPO adapter to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DPO Model using Key-Only Preference Pairs")

    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV containing Judge results")
    parser.add_argument("--base_model", type=str, default="unsloth/gemma-2b-it-bnb-4bit")
    parser.add_argument("--output_dir", type=str, default="models/dpo_adapter_keyonly")
    parser.add_argument("--max_seq_length", type=int, default=4096)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    main(args)