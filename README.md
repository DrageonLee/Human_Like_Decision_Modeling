# Human-Like Decision Modeling with Small LLMs

A research project investigating whether **Small Language Models (SLMs)** can replicate human-like decision-making patterns in cognitive tasks. We utilize **Key-Only Direct Preference Optimization (DPO)** to align reasoning processes with human choices while adhering to strict evaluation constraints.

## Goal
To bridge the gap between large-scale cognitive models (e.g., Centaur, 70B) and accessible SLMs by leveraging **semantic similarity cues** and **indirect reasoning alignment** to predict human behavior in the Odd-One-Out task.

## Dataset
* **Source:** Psych-101 Dataset (Hebart 2023 subset)
* **Task:** Odd-One-Out (Triplets of objects)
* **Format:** "A: item1, B: item2, C: item3. You press <<?>>"
* **Scale:** 10,000 training samples used for preference construction

## Methodology

### Models & Approaches
* **Baselines:**
    * **M1 (Symbolic SFT):** Standard fine-tuning on key labels (A/B/C).
    * **M3 (Semantic Analysis):** Using Sentence-BERT embeddings to predict choices based on semantic distance margins.
* **Proposed Solution:**
    * **M5 (Key-Only DPO):** A novel approach where the model generates reasoning traces during training to form preference pairs (Chosen vs. Rejected), but is optimized to output only the final decision key. This "internalizes" the reasoning benefit without breaking NLL evaluation compatibility.

### Pipeline Architecture
1.  **Reasoning Generation:** Base Gemma-2B generates dual reasoning paths (Prediction A vs. B).
2.  **Preference Judging:** An external LLM Judge (Claude) evaluates reasoning coherence and correctness to assign "Chosen" and "Rejected" labels.
3.  **DPO Training:** Fine-tuning the model to prefer the "Chosen" key using LoRA adapters.

## Results
The Key-Only DPO model demonstrated measurable alignment improvements over the baseline, confirming that preference learning can distill reasoning capabilities into small models.

| Model | NLL (Lower is better) | Key Findings |
|-------|----------------------|--------------|
| **Base (Gemma-2B)** | 5.349 | Noisy baseline reasoning limits zero-shot performance. |
| **DPO Aligned** | **5.203** | **-0.146 improvement**; successfully captures human-like preferences indirectly. |

## Key Insights
* **Semantic Margins Matter:** SBERT similarity is a strong predictor of human choices in "high-margin" (easy) trials but struggles with nuanced decisions.
* **Indirect Reasoning:** DPO effectively serves as a mechanism to incorporate reasoning benefits into a model that must output simple keys at inference time.
* **Resource Efficiency:** Meaningful cognitive alignment is possible on consumer-grade hardware (e.g., Colab T4/A100) using 2B parameter models.

## Tech Stack
* **Core:** Python, PyTorch
* **LLM Frameworks:** Hugging Face Transformers, Unsloth (for efficient fine-tuning)
* **Alignment:** TRL (Transformer Reinforcement Learning), PEFT (LoRA/QLoRA)
* **Analysis:** Sentence-Transformers (SBERT), Scikit-learn (t-SNE)

## Project Structure
```text
HUMAN-LIKE-DECISION-MODELING
├── data
│   ├── raw                  # Original Psych-101 data
│   └── processed            # Generated reasoning & Judged DPO pairs
├── models
│   └── dpo_model_final      # Saved LoRA adapters
├── notebooks
│   └── 01_analysis_sbert.ipynb  # SBERT embedding & Margin analysis
├── src
│   ├── utils.py             # Shared utility functions (prompting, parsing)
│   ├── generate_reasoning.py # 1. Inference script for reasoning traces
│   ├── train_dpo.py         # 2. Training script for Key-Only DPO
│   └── evaluate_model.py    # 3. NLL Evaluation script
├── requirements.txt
└── README.md
```

## Future Work
* **SFT + DPO**: Apply DPO on top of a Supervised Fine-Tuned (SFT) model to reduce initial reasoning noise.
* **Data Scaling**: Expand preference dataset size beyond the current subset.
* **Multi-Judge Aggregation**: Use multiple LLM judges to reduce noise in preference labels.

## Detailed Report
[View Full Project Report PDF](./report/NLP_2025_Fall_Final_Report.pdf)