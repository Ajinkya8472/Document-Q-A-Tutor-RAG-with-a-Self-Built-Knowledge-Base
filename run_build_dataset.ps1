# Step by step (run each line, wait for it to finish):
#   1) python build_corpus.py                       (internet; HF offline NOT needed)
#   2) this script: generate the Q&A set with the local LLM
$env:HF_HOME="D:\hf_cache"
$env:HF_HUB_OFFLINE="1"
$env:HF_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
$env:BACKEND="hf"
Set-Location "D:\AI\Jury-2 GEN AI"
.\venv\Scripts\Activate.ps1
python make_qa.py corpus qa_dataset.json 400