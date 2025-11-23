# MiniLLM — Quick Start Guide (FT8 Big Model)

MiniLLM is a tiny educational transformer model designed to help you understand LLM structure in an FT8 context.  
It lets you train a miniature GPT-style model on your FT8 log files and generate realistic CQ/Reply/QSO flows (more or less).

If you have any questions about the MiniLLM code or training process, simply ask **ChatGPT** — this project is fully ChatGPT-friendly.

---

🚀 1. Install Dependencies

Create a virtual environment (optional but recommended):

python -m venv .venv
.\.venv\Scripts\activate
Install dependencies:

pip install -r requirements.txt
Verify GPU (optional):
python -m mini_llm.gpu_check

📁 2. Prepare the FT8 Corpus
Place your raw FT8 logs here:
data/raw/RxTxLog.txt

Process the raw log into a training corpus:
python scripts/process_RxTxLog.py
This generates: 
data/processed/ft8_corpus.txt
This file is used for training.

🧠 3. Train the Big FT8 Model
Train the transformer on your FT8 corpus:
python -m mini_llm.train_ft8_big
This produces:
phase4_ft8_big_model.pt
Training takes roughly 40 seconds per epoch on an RTX 2070.

✨ 4. Generate FT8-style Messages
Use the big sampler:
python -m mini_llm.sample_ft8_big
This produces plausible FT8-style traffic — CQ calls, signal reports, RR73, etc.

🎯 5. Extract a QSO Flow
To extract a structured 5-stage FT8 exchange from generated text:  
python -m mini_llm.qso_ft8_big  
Example output (your results will vary):  
============== RAW GENERATED TEXT ==============  
CQ AG6AQ CM97  
K7TCH AG6AQ -02  
AG6AQ KF0X R-09  
AG6AQ KC5SM RR73  
KI5CVU N5VAN 73  
...  
============== EXTRACTED QSO ================  
[CQ]       CQ AG6AQ CM97  
[REPLY]    K7TCH AG6AQ -2  
[R-REPORT] AG6AQ KF0X R-09  
[RR73]     AG6AQ KC5SM RR73  
[73]       KI5CVU N5VAN 73  
Because of its small scale, this mini-LLM has difficulty generating fully authentic FT8 messages, but it captures the core structure well. You can fine-tune or extend it if you're interested.  

🙋 Need Help?  
MiniLLM is intentionally simple.  

If you have any questions — training, dataset preparation, code behavior, FT8 logic, or customizing the model:  

👉 Ask ChatGPT directly — all MiniLLM files are readable and ChatGPT-compatible.  

Enjoy experimenting with your tiny FT8 transformer!
