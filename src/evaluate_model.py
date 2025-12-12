import argparse
import math
import re
import pandas as pd
import torch
import os
from datasets import load_dataset
from unsloth import FastLanguageModel
from peft import PeftModel  # Explicit import for adapter loading
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict

# Regex pattern to extract the quiz body and the ground truth answer
# Matches format like: "B: prune, J: nail polish, and K: diskette. You press <<K>>"
QUIZ_GT_PATTERN = r'([A-Z]:\s+[^,]+,\s+[A-Z]:\s+[^,]+,\s+and\s+[A-Z]:\s+[^.]+)\.\s+You press\s+<<([A-Z])>>'

# Regex to extract available option letters (e.g., A, B, C) from the quiz question
OPTION_LETTER_PATTERN = r'([A-Z]):'


def extract_quiz_and_gt_from_prompt(prompt_text: str) -> List[Tuple[str, str]]:
    """
    Parses the raw text to extract (quiz_question, ground_truth) pairs.
    """
    matches = re.findall(QUIZ_GT_PATTERN, prompt_text)
    pairs = []
    for quiz_body, gt in matches:
        pairs.append((quiz_body.strip() + ".", gt.strip()))
    return pairs


def extract_option_letters(quiz_question: str) -> List[str]:
    """
    Extracts the option letters (Keys) from the quiz question.
    e.g., "A: apple, B: banana..." -> ['A', 'B']
    """
    letters = re.findall(OPTION_LETTER_PATTERN, quiz_question)
    uniq = []
    for x in letters:
        if x not in uniq:
            uniq.append(x)
    return uniq


def default_keyonly_prompt(quiz_question: str) -> str:
    """
    Constructs the minimal key-only prompt used for Centaur-style evaluation.
    """
    return (
        "Please indicate which object is the odd one out of three objects.\n"
        "Answer with a single letter only.\n\n"
        f"{quiz_question}\n"
        "Answer:"
    )


def token_id_for_letter(tokenizer, letter: str) -> Optional[int]:
    """
    Finds the token ID for a single capital letter.
    Handles tokenizers that might prepend a space (e.g., Llama).
    """
    ids = tokenizer.encode(letter, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    
    ids2 = tokenizer.encode(" " + letter, add_special_tokens=False)
    if len(ids2) == 1:
        return ids2[0]
        
    return None


@torch.no_grad()
def nll_for_one(prompt: str, correct_letter: str, option_letters: List[str], model, tokenizer) -> Optional[float]:
    """
    Calculates the Negative Log-Likelihood (NLL) for a single example.
    The probability is normalized over the valid option letters (3-way evaluation).
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    next_token_logits = outputs.logits[0, -1, :]

    option_ids = []
    for letter in option_letters:
        tid = token_id_for_letter(tokenizer, letter)
        if tid is None:
            return None
        option_ids.append(tid)

    correct_id = token_id_for_letter(tokenizer, correct_letter)
    if correct_id is None or correct_id not in option_ids:
        return None

    relevant_logits = torch.stack([next_token_logits[i] for i in option_ids], dim=0)
    log_probs = torch.log_softmax(relevant_logits, dim=0)
    
    correct_index = option_ids.index(correct_id)
    nll = -log_probs[correct_index].item()
    
    return nll


def main():
    parser = argparse.ArgumentParser(description="Evaluate Model NLL on Psych-101 Dataset")
    
    # Updated Arguments structure based on your request
    parser.add_argument(
        "--base_model",
        type=str,
        default="unsloth/gemma-2b-it-bnb-4bit",
        help="Base model name or path (e.g., unsloth/gemma-2b-it-bnb-4bit)"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to PEFT/DPO adapter (optional). If None, evaluates the base model."
    )
    
    parser.add_argument("--fold", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--task_name", type=str, default="hebart2023things", help="Experiment name prefix")
    parser.add_argument("--max_subjects", type=int, default=50, help="Number of subjects")
    parser.add_argument("--max_quizzes_per_subject", type=int, default=50)
    parser.add_argument("--output_file", type=str, default="results/evaluation_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # --- 1. Load Data ---
    print(f"Loading dataset split: {args.fold}...")
    train_ds = load_dataset("marcelbinz/Psych-101")
    test_ds = load_dataset("marcelbinz/Psych-101-test")
    dataset = train_ds if args.fold == "train" else test_ds

    filtered_dataset = dataset[args.fold].filter(lambda ex: ex["experiment"].startswith(args.task_name))
    print(f"Found {len(filtered_dataset)} subjects for task '{args.task_name}'.")

    # --- 2. Load Model (Updated Logic) ---
    print(f"Loading base model: {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=5000,
        dtype=None,
        load_in_4bit=True,
    )

    # Gemma-specific tokenizer setting
    if "gemma" in args.base_model.lower():
        tokenizer.padding_side = "right"

    # Explicitly load adapter if provided
    if args.adapter_path is not None:
        print(f"Loading adapter from: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    FastLanguageModel.for_inference(model)
    model.eval()

    # --- 3. Evaluation Loop ---
    total_nll = 0.0
    count = 0
    skipped_count = 0

    num_subjects_to_process = min(args.max_subjects, len(filtered_dataset))
    print(f"Evaluating on {num_subjects_to_process} subjects...")

    for i in tqdm(range(num_subjects_to_process), desc="Evaluating"):
        text_data = filtered_dataset[i]["text"]
        quiz_pairs = extract_quiz_and_gt_from_prompt(text_data)
        quiz_pairs = quiz_pairs[: args.max_quizzes_per_subject]

        for quiz, gt in quiz_pairs:
            options = extract_option_letters(quiz)
            if len(options) < 3:
                skipped_count += 1
                continue

            prompt = default_keyonly_prompt(quiz)
            nll = nll_for_one(prompt, gt, options[:3], model, tokenizer)
            
            if nll is None or math.isnan(nll) or math.isinf(nll):
                skipped_count += 1
                continue

            total_nll += nll
            count += 1

    # --- 4. Save Results ---
    avg_nll = total_nll / max(count, 1)
    
    print("=" * 40)
    print(f"Evaluation Complete")
    print(f"Base Model: {args.base_model}")
    print(f"Adapter: {args.adapter_path if args.adapter_path else 'None'}")
    print(f"Total evaluated items: {count}")
    print(f"Average NLL (3-way normalized): {avg_nll:.6f}")
    print("=" * 40)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    results_df = pd.DataFrame({
        "base_model": [args.base_model],
        "adapter_path": [args.adapter_path],
        "avg_nll": [avg_nll],
        "evaluated_count": [count],
        "fold": [args.fold]
    })
    
    # If file exists, append; otherwise create new
    if os.path.exists(args.output_file):
        results_df.to_csv(args.output_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(args.output_file, index=False)
        
    print(f"Results appended to {args.output_file}")

if __name__ == "__main__":
    main()