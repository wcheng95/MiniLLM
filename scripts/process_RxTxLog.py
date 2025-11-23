# scripts/process_RxTxLog.py
#
# Build a clean FT8 message corpus from RxTxLog.txt
#
# PRE-FILTER RULES:
# 1) Remove the line if it does NOT start with 'T' or 'R' (uppercase only)
# 2) Remove the first 28 characters from the line
# 3) Remove the line if the remaining part contains NO letters
#
# FT8 RULES:
# - If a line begins with "CQ <1-4 letters>", treat it as ONE field (preserve the space)
# - A valid FT8 message has at least 3 fields
# - If a line has more than 3 fields, keep only the first 3
# - Remove lines that reduce to exactly two numeric fields
# - Remove malformed lines
# - Remove only adjacent duplicates (global duplicates allowed)
#
# Input : data/raw/RxTxLog.txt
# Output: data/processed/ft8_corpus.txt

import os
import re
import random

INPUT_FILE  = "data/raw/RxTxLog.txt"
OUTPUT_FILE = "data/processed/ft8_corpus.txt"

INT_RE = re.compile(r"^[+-]?\d+$")
CQ_LETTERS_RE = re.compile(r"^[A-Za-z]{1,4}$")
CALLSIGN_RE = re.compile(r"\b[A-Z0-9]{3,6}\b")

# ---- CONFIG ----
DOWN_SAMPLE_CQ = 0.20   # keep only 20% of CQ lines (tunable)
MIN_QSO_NEIGHBORS = 1   # require neighbors for non-CQ lines


def is_two_numbers_only(text: str) -> bool:
    t = text.split()
    return len(t) == 2 and all(INT_RE.match(x) for x in t)


def clean_ft8_line(s: str):
    """Return cleaned 3-field FT8 message or None."""
    tokens = s.split()
    if len(tokens) < 3:
        return None

    # CQ-expansion (CQ + 1–4 letters = one field)
    if tokens[0].upper() == "CQ" and CQ_LETTERS_RE.match(tokens[1]):
        if len(tokens) < 4:
            return None
        f1 = tokens[0] + " " + tokens[1]
        f2 = tokens[2]
        f3 = tokens[3]
    else:
        f1, f2, f3 = tokens[0], tokens[1], tokens[2]

    msg = f"{f1} {f2} {f3}".strip()
    if is_two_numbers_only(msg):
        return None

    # must contain letters
    if not any(c.isalpha() for c in msg):
        return None

    return msg


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] File not found: {INPUT_FILE}")
        return

    print("[INFO] Reading:", INPUT_FILE)

    cleaned = []
    last_was_cq = False

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.rstrip("\n")

            # Must begin with T or R
            if not raw.startswith(("T", "R")):
                continue

            # remove first 28 chars
            if len(raw) < 28:
                continue
            s = raw[28:].strip()

            # must contain letters
            if not any(c.isalpha() for c in s):
                continue

            # clean FT8 form
            msg = clean_ft8_line(s)
            if msg is None:
                continue

            # collapse CQ clusters
            if msg.startswith("CQ "):
                if last_was_cq:
                    continue  # skip duplicate CQ in cluster
                last_was_cq = True
            else:
                last_was_cq = False

            cleaned.append(msg)

    print("[INFO] After initial clean:", len(cleaned))

    # ---- Downsample CQ heavy bias ----
    final_lines = []
    for msg in cleaned:
        if msg.startswith("CQ "):
            if random.random() < DOWN_SAMPLE_CQ:
                final_lines.append(msg)
        else:
            final_lines.append(msg)

    print("[INFO] After CQ downsampling:", len(final_lines))

    # ---- Deduplicate (global) ----
    final_lines = list(dict.fromkeys(final_lines))
    print("[INFO] After global dedupe:", len(final_lines))

    # ---- Shuffle (for more uniform training) ----
    random.shuffle(final_lines)

    # ---- Save ----
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for line in final_lines:
            out.write(line + "\n")

    print(f"[OK] Saved balanced corpus: {OUTPUT_FILE}")
    print(f"[OK] Total lines: {len(final_lines)}")


if __name__ == "__main__":
    main()