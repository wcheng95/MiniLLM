# mini_llm/qso_ft8.py

import torch
from mini_llm.tokenizer import ByteTokenizer
from mini_llm.model import TinyGPT
from mini_llm.config import Config
import re

def sample_with_filters(
    model,
    tokenizer,
    start_text,
    max_new_tokens,
    block_size,
    device,
    my_call="AG6AQ",
    temperature=0.7,
    top_k=50,
    top_p=0.90,
):
    model.eval()
    with torch.no_grad():
        idx = torch.tensor([tokenizer.encode(start_text)], dtype=torch.long).to(device)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]

            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]

            logits = logits / temperature

            probs = torch.softmax(logits, dim=-1).squeeze(0)
            # --- Bias toward emitting our own callsign ---

            # Top-k
            if top_k is not None:
                values, _ = torch.topk(probs, top_k)
                threshold = values.min()
                probs = torch.where(probs < threshold, torch.zeros_like(probs), probs)

            # Top-p (nucleus)
            if top_p is not None:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=0)
                mask = cumulative <= top_p
                mask[0] = True  # always keep best token
                probs = torch.zeros_like(probs).scatter(0, sorted_idx[mask], sorted_probs[mask])

            probs = probs / probs.sum()

            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id.unsqueeze(0)], dim=1)

    return tokenizer.decode(idx[0].tolist())


def classify_line(line: str):
    s = line.strip().upper()
    if not s:
        return "EMPTY"

    if "CQ POTA" in s or "CQ SOTA" in s:
        return "CQ_POTA"
    if s.startswith("CQ "):
        return "CQ"

    if "RR73" in s or " 73" in s:
        return "73"

    if " R-" in s or s.endswith(" R-") or " R-" in s:
        return "R_REPORT"

    if any(rep in s for rep in [" R-", " R+", " -", "+"]) and any(ch.isdigit() for ch in s):
        return "REPORT"

    return "OTHER"


def generate_qso(model, tokenizer, my_call="AG6AQ", my_grid="CM97", device=None):
    """
    Generate a pseudo-stateful FT8 QSO:
    - Start with CQ MYCALL MYGRID
    - Let the model produce multiple lines
    - Pick lines for CQ, reply, R-report, and 73 if possible
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompt = f"CQ {my_call} {my_grid} "
    
    text = sample_with_filters(
        model,
        tokenizer,
        start_text=prompt,
        max_new_tokens=600,
        block_size=128,
        device=device,
        my_call=my_call,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
    )

    lines = [ln for ln in text.splitlines() if ln.strip()]

    CALLSIGN_RE = re.compile(r"\b[A-Z0-9]{3,6}\b")

    def contains_other_call(line, my_call):
        s = line.upper()

        # Extract callsigns using regex
        calls = CALLSIGN_RE.findall(s)

        # No calls → cannot be reply
        if not calls:
            return False

        # If MY call not present → not reply to me
        if my_call not in calls:
            return False

        # If there is ANY other call in the line → this is a reply
        return len(calls) >= 2

    cq_line = None
    reply_line = None
    r_report_line = None
    final_73_line = None

    for ln in lines:
        cls = classify_line(ln)
        up = ln.upper()

        # try to pick:
        # - CQ line that has our call
        # - reply line that has our call and some other call
        # - R-report line that has our call
        # - 73 line
        if cq_line is None and "CQ" in up and my_call in up:
            cq_line = ln
        elif reply_line is None and contains_other_call(ln, my_call) and not ln.upper().startswith(f"CQ {my_call}"):
            reply_line = ln
        elif reply_line is None and my_call in up and not up.startswith(f"CQ {my_call}"):
            reply_line = ln

        elif r_report_line is None and my_call in up and cls == "R_REPORT":
            r_report_line = ln
        elif final_73_line is None and cls == "73":
            final_73_line = ln

    qso_lines = []
    if cq_line:
        qso_lines.append(("CQ", cq_line))
    if reply_line:
        qso_lines.append(("REPLY", reply_line))
    if r_report_line:
        qso_lines.append(("R-REPORT", r_report_line))
    if final_73_line:
        qso_lines.append(("73", final_73_line))

    return qso_lines, text


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Config()
    cfg.vocab_size = 256
    cfg.block_size = 128
    cfg.n_embd = 128
    cfg.n_head = 4
    cfg.n_layer = 2
    cfg.dropout = 0.1

    tokenizer = ByteTokenizer()
    model = TinyGPT(cfg).to(device)

    state_dict = torch.load("phase2_ft8_model.pt", weights_only=True, map_location=device)
    model.load_state_dict(state_dict)

    my_call = "AG6AQ"
    my_grid = "CM97"

    qso_lines, raw_text = generate_qso(model, tokenizer, my_call=my_call, my_grid=my_grid, device=device)

    print("============== RAW GENERATED TEXT ==============")
    print(raw_text)
    print("============== EXTRACTED QSO FLOW ==============")
    if not qso_lines:
        print("(Could not extract a QSO-like sequence this time; run again.)")
    else:
        for role, line in qso_lines:
            print(f"[{role}] {line}")


if __name__ == "__main__":
    main()
