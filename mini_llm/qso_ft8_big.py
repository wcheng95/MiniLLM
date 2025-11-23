# mini_llm/qso_ft8_big.py
# FINAL version WITHOUT partner-lock + strict FT8 field rule

import torch
import re

from mini_llm.tokenizer import ByteTokenizer
from mini_llm.model import TinyGPT
from mini_llm.config import Config


# ---------------------------------------------------------
# FT8 field rule helper
# ---------------------------------------------------------
def allowed_field_count(tokens):
    # Only CQ <TAG> <CALL> <GRID> may have 4 fields
    if len(tokens) >= 2 and tokens[0] == "CQ":
        tag = tokens[1]
        if tag.isalpha() and tag.isupper() and 1 <= len(tag) <= 4:
            return 4
    return 3


# ---------------------------------------------------------
# Sampling with FT8 field enforcement
# ---------------------------------------------------------
def sample_with_filters(
    model,
    tokenizer,
    start_text,
    max_new_tokens,
    block_size,
    device,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
):
    model.eval()
    with torch.no_grad():
        idx = torch.tensor([tokenizer.encode(start_text)], dtype=torch.long).to(device)

        newline_id = tokenizer.encode("\n")[0]

        for _ in range(max_new_tokens):

            # context
            idx_cond = idx[:, -block_size:]

            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1).squeeze(0)

            # top-k
            if top_k is not None:
                values, _ = torch.topk(probs, top_k)
                thresh = values.min()
                probs = torch.where(probs < thresh, torch.zeros_like(probs), probs)

            # top-p
            if top_p is not None:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=0)
                mask = cumulative <= top_p
                mask[0] = True
                probs = torch.zeros_like(probs).scatter(0, sorted_idx[mask], sorted_probs[mask])

            probs /= probs.sum()
            next_id = torch.multinomial(probs, 1)

            # add token
            idx = torch.cat([idx, next_id.unsqueeze(0)], dim=1)

    return tokenizer.decode(idx[0].tolist())

def sanitize_ft8_line(ln):
    tokens = ln.split()

    # No need to fix short lines
    if len(tokens) <= 3:
        return ln

    # If 4 tokens: only valid format is CQ <TAG> <CALL> <GRID>
    if len(tokens) == 4:
        if tokens[0] == "CQ":
            tag = tokens[1]
            # TAG: 1–4 uppercase letters
            if tag.isalpha() and tag.isupper() and 1 <= len(tag) <= 4:
                return ln  # valid 4-field CQ TAG message

        # Otherwise → remove the 4th field
        return " ".join(tokens[:3])

    # If more than 4 tokens → truncate to 3
    return " ".join(tokens[:3])

# ---------------------------------------------------------
# FT8 classifier
# ---------------------------------------------------------
def classify_line(line):
    s = line.strip().upper()

    if not s:
        return "EMPTY"

    if s.startswith("CQ "):
        return "CQ"

    # RR73 must be checked before plain 73
    if "RR73" in s:
        return "RR73"

    if s.endswith(" 73") or s.endswith("73"):
        return "73"

    if " R-" in s:
        return "R_REPORT"

    if any(rep in s for rep in [" -", "+"]) and any(c.isdigit() for c in s):
        return "REPORT"

    return "OTHER"


# ---------------------------------------------------------
# Callsign regex
# ---------------------------------------------------------
CALLSIGN_RE = re.compile(r"\b[A-Z0-9/]{2,7}\b")

def extract_calls(line):
    return CALLSIGN_RE.findall(line.upper())


# ---------------------------------------------------------
# QSO extractor WITHOUT partner-lock
# ---------------------------------------------------------
def generate_qso_big(model, tokenizer, my_call="AG6AQ", my_grid="CM97", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate output text
    prompt = f"CQ {my_call} {my_grid} "
    text = sample_with_filters(
        model,
        tokenizer,
        start_text=prompt,
        max_new_tokens=1800,
        block_size=256,
        device=device,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
    )

    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Apply 4-field correction
    lines = [sanitize_ft8_line(ln) for ln in raw_lines]

    my_call_up = my_call.upper()

    classified = [
        (classify_line(ln), ln)
        for ln in lines
        if my_call_up in ln.upper()
]

    # -------------------
    # Extract steps
    # -------------------

    # 1. CQ line
    cq = next((ln for cls, ln in classified if cls == "CQ"), None)
    if cq:
        parts = cq.split()
        allowed = allowed_field_count(parts)
        cq = " ".join(parts[:allowed])

    # 2. First reply (any reply that is not CQ)
    reply = next((ln for cls, ln in classified if cls == "REPORT" and ln != cq), None)

    # 3. R-report (AG6AQ may or may not appear)
    r_report = next((ln for cls, ln in classified if cls == "R_REPORT"), None)

    # 4. 73
    rr73 = next((ln for cls, ln in classified if cls == "73"), None)

    # -------------------
    # Build QSO list
    # -------------------
    qso = []
    if cq: qso.append(("CQ", cq))
    if reply: qso.append(("REPLY", reply))
    if r_report: qso.append(("R-REPORT", r_report))
    if rr73: qso.append(("73", rr73))

    return qso, text


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    cfg = Config()
    cfg.vocab_size = 256
    cfg.block_size = 256
    cfg.n_embd = 256
    cfg.n_head = 4
    cfg.n_layer = 4
    cfg.dropout = 0.1

    tokenizer = ByteTokenizer()
    model = TinyGPT(cfg).to(device)

    state_dict = torch.load("phase4_ft8_big_model.pt", weights_only=True, map_location=device)
    model.load_state_dict(state_dict)

    qso, raw = generate_qso_big(model, tokenizer, "AG6AQ", "CM97", device)

    print("============== RAW GENERATED TEXT ==============")
    print(raw)
    print("================ EXTRACTED QSO ================")
    if not qso:
        print("(No QSO extracted)")
    else:
        for role, line in qso:
            print(f"[{role}] {line}")


if __name__ == "__main__":
    main()
