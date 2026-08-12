"""Build Financial Planning SFT Training & Evaluation Datasets from EPUB.

Extracts chunks from 'Psychology of Financial Plannin - Brad Klontz.epub' and generates:
1. data/financial_planning/training_data.jsonl (SFT QA pairs for training)
2. data/financial_planning/evaluation_data.jsonl (20 domain evaluation prompts)
"""

import json
import re
from pathlib import Path
from gnn_experiment.datagen.epub import chunk_epub_file

REPO_ROOT = Path(__file__).resolve().parent.parent
FINANCIAL_DIR = REPO_ROOT / "data" / "financial_planning"
EPUB_PATH = FINANCIAL_DIR / "Psychology of Financial Plannin - Brad Klontz.epub"


def extract_qa_from_text(chunk_text: str, source_path: str, heading_path: list[str]) -> list[dict]:
    """Extracts clean financial planning Q&A pairs from text chunks."""
    records = []
    text = chunk_text.strip()
    if len(text) < 150:
        return records

    # Split into paragraphs or major sections
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]
    
    for idx, p in enumerate(paragraphs[:3]):
        # Formulate structured questions and expert answers based on financial psychology & planning concepts
        lines = [line.strip() for line in p.split("\n") if line.strip()]
        clean_p = " ".join(lines)
        if len(clean_p) < 100:
            continue

        # Extract key concept or thesis from paragraph
        heading_context = " / ".join(heading_path) if heading_path else "Financial Planning Principles"
        
        # Formulate prompt and answer
        user_prompt = f"In financial planning and behavioral finance ({heading_context}), explain the core principles of: {clean_p[:80]}..."
        assistant_answer = f"According to the Psychology of Financial Planning, key principles include:\n\n{clean_p}"

        rec = {
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_answer},
            ],
            "meta": {
                "tool": "Psychology of Financial Planning",
                "source_path": source_path,
                "heading_path": heading_path,
                "question_gen_version": "v1",
                "answer_gen_version": "v1",
            },
            "text": f"### Question:\n{user_prompt}\n\n### Answer:\n{assistant_answer}",
        }
        records.append(rec)
        
    return records


def build_datasets():
    FINANCIAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading EPUB from {EPUB_PATH}...")
    chunks = chunk_epub_file(EPUB_PATH, tool="financial_planning", chunk_size_chars=1800, min_chars=120)
    print(f"Extracted {len(chunks)} raw text chunks from EPUB.")

    training_records = []
    for chunk in chunks:
        qas = extract_qa_from_text(chunk.text, chunk.source_path, chunk.heading_path)
        training_records.extend(qas)

    sft_out_path = FINANCIAL_DIR / "training_data.jsonl"
    with open(sft_out_path, "w", encoding="utf-8") as f:
        for rec in training_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(training_records)} SFT training records to {sft_out_path}")

    # Build 20 rigorous evaluation prompts
    eval_prompts = [
        {"id": "fin_01", "category": "behavioral_finance", "prompt": "What are money scripts in financial therapy, and how do money avoidance and money status scripts affect client savings behaviors?"},
        {"id": "fin_02", "category": "behavioral_finance", "prompt": "How should a financial planner address loss aversion when a client wants to panic-sell during market volatility?"},
        {"id": "fin_03", "category": "behavioral_finance", "prompt": "What is financial flashpoint analysis, and how do childhood experiences shape adult financial decision-making?"},
        {"id": "fin_04", "category": "behavioral_finance", "prompt": "How does confirmation bias impact client investment choices, and what techniques can a planner use to counteract it?"},
        {"id": "fin_05", "category": "behavioral_finance", "prompt": "Explain the difference between financial literacy and financial psychology in client outcomes."},
        
        {"id": "fin_06", "category": "client_communication", "prompt": "How can a financial planner use active listening and open-ended questions to uncover a client's core financial values?"},
        {"id": "fin_07", "category": "client_communication", "prompt": "What strategies help a financial planner navigate money conflict between married couples during joint planning sessions?"},
        {"id": "fin_08", "category": "client_communication", "prompt": "How should a planner deliver disappointing news, such as a client being unable to retire at their target age?"},
        {"id": "fin_09", "category": "client_communication", "prompt": "What is motivational interviewing in financial counseling, and how does it encourage long-term behavioral change?"},
        {"id": "fin_10", "category": "client_communication", "prompt": "How can a planner establish trust and rapport with a new client who feels anxious or ashamed about debt?"},

        {"id": "fin_11", "category": "risk_management", "prompt": "What is the distinction between a client's risk tolerance, risk capacity, and risk perception?"},
        {"id": "fin_12", "category": "risk_management", "prompt": "How does recency bias distort a client's perceived risk during extended bull or bear markets?"},
        {"id": "fin_13", "category": "risk_management", "prompt": "Why is sequence of returns risk critical in retirement income planning, and how can it be mitigated?"},
        {"id": "fin_14", "category": "risk_management", "prompt": "How should a financial planner handle mental accounting bias when clients treat inherited money differently from earned income?"},
        {"id": "fin_15", "category": "risk_management", "prompt": "What role does emotional intelligence play in managing portfolio risk during economic crises?"},

        {"id": "fin_16", "category": "anti_pattern_correction", "prompt": "Should a financial planner focus strictly on mathematical portfolio optimization without addressing client emotional reactions?"},
        {"id": "fin_17", "category": "anti_pattern_correction", "prompt": "Is giving clients comprehensive financial education alone sufficient to change destructive spending habits?"},
        {"id": "fin_18", "category": "anti_pattern_correction", "prompt": "Should a planner assume a client's risk tolerance questionnaire score remains constant throughout their lifetime?"},
        {"id": "fin_19", "category": "anti_pattern_correction", "prompt": "Is it best to immediately challenge and correct a client's irrational financial beliefs during the initial consultation?"},
        {"id": "fin_20", "category": "anti_pattern_correction", "prompt": "Should financial goals be established solely around numerical target returns rather than personal life values?"},
    ]

    eval_out_path = FINANCIAL_DIR / "evaluation_data.jsonl"
    with open(eval_out_path, "w", encoding="utf-8") as f:
        for prompt in eval_prompts:
            f.write(json.dumps(prompt) + "\n")

    print(f"Wrote {len(eval_prompts)} domain evaluation prompts to {eval_out_path}")


if __name__ == "__main__":
    build_datasets()
