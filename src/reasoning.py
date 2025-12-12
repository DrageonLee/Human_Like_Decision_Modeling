import os
import argparse
import torch
import numpy as np
import pandas as pd
import random
from datasets import load_dataset
from unsloth import FastLanguageModel
from tqdm import tqdm

# Import shared utility functions from utils.py
# Note: 'create_reasoning_prompt' is used here for data generation
from utils import (
    extract_quiz_and_gt_from_prompt,
    create_reasoning_prompt,
    extract_prediction_and_reasoning
)

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
    """
    Generates a text response from the LLM based on the given prompt.
    """
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature,
            use_cache=True,
            do_sample=True
        )
    # Decode and remove special tokens
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


def process_subject_dual(model, tokenizer, prompt_text: str, participant_id: int) -> pd.DataFrame:
    """
    Processes a single subject's data by generating two distinct reasoning paths (A and B)
    for each quiz question found in the text.

    Args:
        model: The loaded LLM.
        tokenizer: The loaded tokenizer.
        prompt_text (str): The raw text containing quiz questions for one subject.
        participant_id (int): The ID of the participant.

    Returns:
        pd.DataFrame: A dataframe containing the generated pairs (PredictionA/B, ReasoningA/B).
    """
    # 1. Parse questions and ground truths from the raw text
    quiz_gt_pairs = extract_quiz_and_gt_from_prompt(prompt_text)
    results = []

    for idx, (question, gt) in enumerate(quiz_gt_pairs):
        # 2. Create the input prompt for the model
        # Using 'create_reasoning_prompt' to instruct the model to output reasoning
        model_input = create_reasoning_prompt(question)
        
        # --- Generate Response A (Deterministic Seed) ---
        seed_a = 42 + idx
        random.seed(seed_a)
        torch.manual_seed(seed_a)
        
        # Generate and parse A
        resp_a = generate_response(model, tokenizer, model_input, temperature=0.7)
        pred_a, reas_a = extract_prediction_and_reasoning(resp_a)
        
        # --- Generate Response B (Different Seed & Higher Temp) ---
        # We increase temperature slightly to encourage diversity for DPO pairs
        seed_b = 12345 + idx
        random.seed(seed_b)
        torch.manual_seed(seed_b)
        
        # Generate and parse B
        resp_b = generate_response(model, tokenizer, model_input, temperature=0.85)
        pred_b, reas_b = extract_prediction_and_reasoning(resp_b)

        # 3. Collect result
        results.append({
            "participant": participant_id,
            "Quiz": question,
            "GT": gt,
            "PredictionA": pred_a,
            "ReasoningA": reas_a,
            "PredictionB": pred_b,
            "ReasoningB": reas_b,
            # Optional: Save raw responses for debugging
            # "RawA": resp_a,
            # "RawB": resp_b
        })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Generate Reasoning Traces (A/B) for DPO")
    
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-2b-it-bnb-4bit",
                        help="HuggingFace model ID or path")
    parser.add_argument("--max_seq_length", type=int, default=5000,
                        help="Max sequence length for the model context")
    parser.add_argument("--max_subjects", type=int, default=10,
                        help="Maximum number of subjects to process from the dataset")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Directory to save the output CSV")
    parser.add_argument("--output_file", type=str, default="raw_reasoning_traces.csv",
                        help="Filename for the output CSV")
    
    args = parser.parse_args()
    
    # --- 1. Load Model ---
    print(f"Loading model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Gemma-specific padding fix
    if "gemma" in args.model_name.lower():
        tokenizer.padding_side = "right"

    FastLanguageModel.for_inference(model)

    # --- 2. Load Dataset ---
    print("Loading Psych-101 dataset (Train split)...")
    dataset = load_dataset("marcelbinz/Psych-101", split='train')
    
    # Filter for 'hebart2023things' experiment
    hebart_dataset = dataset.filter(lambda x: x['experiment'].startswith("hebart2023things"))
    print(f"Total relevant subjects found: {len(hebart_dataset)}")
    
    # --- 3. Process Subjects ---
    all_dataframes = []
    num_to_process = min(args.max_subjects, len(hebart_dataset))
    
    print(f"Processing {num_to_process} subjects to generate reasoning pairs...")
    
    for idx in tqdm(range(num_to_process), desc="Generating"):
        sample = hebart_dataset[idx]
        try:
            df = process_subject_dual(model, tokenizer, sample['text'], sample['participant'])
            if not df.empty:
                all_dataframes.append(df)
        except Exception as e:
            print(f"Error processing participant {sample['participant']}: {e}")
            continue

    # --- 4. Save Results ---
    if all_dataframes:
        final_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, args.output_file)
        
        final_df.to_csv(output_path, index=False)
        print("=" * 40)
        print(f"Generation Complete!")
        print(f"Total questions processed: {len(final_df)}")
        print(f"Results saved to: {output_path}")
        print("=" * 40)
    else:
        print("No data was generated. Please check the dataset or model.")

if __name__ == "__main__":
    main()