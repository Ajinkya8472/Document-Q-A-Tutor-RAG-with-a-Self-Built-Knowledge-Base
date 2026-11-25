# Usage:  .\run_benchmark.ps1 [docs] [questions] [backend]
#   .\run_benchmark.ps1 lora_paper.pdf eval_questions_lora.json hf
#   .\run_benchmark.ps1 corpus qa_dataset_reviewed.json extractive     (no-LLM baseline)
#   .\run_benchmark.ps1 corpus qa_dataset_reviewed.json hf
param([string]$docs="lora_paper.pdf", [string]$qs="eval_questions_lora.json", [string]$backend="hf")
$env:HF_HOME="D:\hf_cache"
$env:HF_HUB_OFFLINE="1"
$env:HF_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
if (-not $env:THRESHOLD) { $env:THRESHOLD="0.25" }
# name result files after the documents so runs never overwrite each other (e.g. results_corpus_hf.md)
$env:TAG = [System.IO.Path]::GetFileNameWithoutExtension($docs)
if ($env:PROMPT) { $env:TAG = $env:TAG + "_" + $env:PROMPT }      # e.g. results_corpus_v2_hf.md
Set-Location "D:\AI\Jury-2 GEN AI"
.\venv\Scripts\Activate.ps1
python benchmark.py $docs $qs $backend
