"""CPU-only data cleaning script for data/financial_planning/training_data.jsonl.

Removes:
1. Informal conversational preambles ("Hey!", "Hey,", "Hey ") from assistant responses.
2. Book/excerpt reference artifacts from user prompts.

Runs 100% on CPU without touching any PyTorch/GPU code.
"""

import json
import re
from pathlib import Path

TRAIN_PATH = Path("data/financial_planning/training_data.jsonl")


def clean_assistant_response(text: str) -> str:
    """Strip informal 'Hey!', 'Hey,', 'Hey ' preambles from assistant text."""
    cleaned = text.strip()
    # Match patterns like: Hey! , Hey, , Hey - , Hey!
    pattern = r"^(hey[\!\,\-\s]+)+"
    match = re.match(pattern, cleaned, flags=re.IGNORECASE)
    if match:
        cleaned = cleaned[match.end() :].strip()
        if cleaned:
            # Capitalize first letter
            cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def clean_user_prompt(prompt: str) -> str:
    """Remove book/excerpt/section reference artifacts from user questions."""
    cleaned = prompt.strip()

    patterns = [
        (r",?\s*according to the [\"'].*?[\"'] section\??", "?"),
        (r",?\s*according to the Klontz book,?", ""),
        (r",?\s*according to the Psychology of Financial Planning\??", "?"),
        (r",?\s*according to the principles discussed in this excerpt\??", "?"),
        (r",?\s*as described in the excerpt,?", ""),
        (r",?\s*as described in the Klontz book,?", ""),
        (r",?\s*as described in the excerpt\??", "?"),
        (r"\s+mentioned in the excerpt\??", "?"),
        (r"\s+discussed in this excerpt\??", "?"),
        (r",?\s*according to the text\??", "?"),
        (r"\s+in the Klontz book\??", "?"),
    ]

    for pat, repl in patterns:
        cleaned = re.sub(pat, repl, cleaned, flags=re.IGNORECASE).strip()

    # Ensure single trailing question mark if original had one
    if prompt.strip().endswith("?") and not cleaned.endswith("?"):
        cleaned = cleaned.rstrip(".") + "?"

    # Clean up double spaces or awkward trailing commas before question marks
    cleaned = re.sub(r",\?", "?", cleaned)
    cleaned = re.sub(r"\s+\?", "?", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)

    return cleaned.strip()


def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"{TRAIN_PATH} does not exist.")

    records = []
    with open(TRAIN_PATH) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records from {TRAIN_PATH}")

    hey_cleaned_count = 0
    prompt_cleaned_count = 0

    cleaned_records = []
    for r in records:
        user_orig = r["messages"][0]["content"]
        ans_orig = r["messages"][1]["content"]

        user_clean = clean_user_prompt(user_orig)
        ans_clean = clean_assistant_response(ans_orig)

        if user_clean != user_orig:
            prompt_cleaned_count += 1
        if ans_clean != ans_orig:
            hey_cleaned_count += 1

        r["messages"][0]["content"] = user_clean
        r["messages"][1]["content"] = ans_clean
        r["text"] = f"### Question:\n{user_clean}\n\n### Answer:\n{ans_clean}"

        cleaned_records.append(r)

    h_pct = 100.0 * hey_cleaned_count / len(records)
    p_pct = 100.0 * prompt_cleaned_count / len(records)
    print(f"Cleaned 'Hey!' preambles from {hey_cleaned_count} assistant responses ({h_pct:.1f}%)")
    print(f"Cleaned book/excerpt references from {prompt_cleaned_count} user prompts ({p_pct:.1f}%)")

    # Overwrite in-place
    with open(TRAIN_PATH, "w") as f:
        for r in cleaned_records:
            f.write(json.dumps(r) + "\n")

    print(f"Successfully updated {TRAIN_PATH} in-place!")


if __name__ == "__main__":
    main()
