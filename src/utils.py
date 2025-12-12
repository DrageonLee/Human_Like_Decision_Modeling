import re
from typing import List, Tuple, Optional

# Matches: "B: prune, J: nail polish, and K: diskette. You press <<K>>"
QUIZ_GT_PATTERN = r'([A-Z]:\s+[^,]+,\s+[A-Z]:\s+[^,]+,\s+and\s+[A-Z]:\s+[^.]+)\.\s+You press\s+<<([A-Z])>>'

# Extract option letters from a quiz line like: "A: apple, B: banana, and C: car."
OPTION_LETTER_PATTERN = r'([A-Z]):'

# Parse model output like:
# "Prediction: A, Reasoning: ..."
# or
# "Prediction: A\nReasoning: ..."
PRED_PATTERN = r'Prediction:\s*([A-Z])'
REASON_PATTERN = r'Reasoning:\s*(.+)'


def extract_quiz_and_gt_from_prompt(prompt_text: str) -> List[Tuple[str, str]]:
    """
    Extracts (quiz_question, ground_truth_key) pairs from Psych-101 subject text.

    Returns:
        quiz_question: e.g. "B: prune, J: nail polish, and K: diskette."
        ground_truth: e.g. "K"
    """
    matches = re.findall(QUIZ_GT_PATTERN, prompt_text)
    pairs: List[Tuple[str, str]] = []
    for quiz_body, gt in matches:
        pairs.append((quiz_body.strip() + ".", gt.strip()))
    return pairs


def extract_option_letters(quiz_question: str) -> List[str]:
    """
    Extracts unique option letters in order from a quiz question.
    """
    letters = re.findall(OPTION_LETTER_PATTERN, quiz_question)
    uniq: List[str] = []
    for x in letters:
        if x not in uniq:
            uniq.append(x)
    return uniq


def create_keyonly_prompt(quiz_question: str) -> str:
    """
    Centaur-style key-only prompt for NLL evaluation.
    Model must output a single letter only.
    """
    return (
        "Please indicate which object is the odd one out of three objects.\n"
        "Answer with a single letter only.\n\n"
        f"{quiz_question}\n"
        "Answer:"
    )


def create_reasoning_prompt(quiz_question: str, base_prompt: Optional[str] = None) -> str:
    """
    Prompt for generating (Prediction, Reasoning) pairs used for preference construction.
    """
    if base_prompt is None:
        base_prompt = (
            "Please indicate which object you think is the odd one out of three objects.\n\n"
            "In other words, choose the object that is the least similar to the other two.\n\n"
            'Respond in the exact format: "Prediction: <LETTER>, Reasoning: <TEXT>".\n'
            "Do not write any other text.\n\n"
        )
    return base_prompt + quiz_question



def extract_prediction_and_reasoning(text: str) -> Tuple[str, str]:
    """
    Robustly extracts Prediction letter and Reasoning text from a model response.

    Returns:
        (prediction_letter, reasoning_text)
        If parsing fails: ("", "")
    """
    m_pred = re.search(PRED_PATTERN, text, flags=re.IGNORECASE)
    if not m_pred:
        return "", ""

    pred = m_pred.group(1).strip().upper()

    m_reason = re.search(REASON_PATTERN, text, flags=re.IGNORECASE | re.DOTALL)
    reasoning = ""
    if m_reason:
        reasoning = " ".join(m_reason.group(1).strip().split())

    return pred, reasoning


def token_id_for_letter(tokenizer, letter: str) -> Optional[int]:
    """
    Finds token id for a single capital letter.
    Tries 'A' then ' A' to handle tokenizers that use leading-space tokens.
    """
    letter = letter.strip().upper()

    ids = tokenizer.encode(letter, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]

    ids2 = tokenizer.encode(" " + letter, add_special_tokens=False)
    if len(ids2) == 1:
        return ids2[0]

    return None
