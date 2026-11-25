# PUBLIC demo: creates a temporary gradio.live link (needs internet; laptop must stay on and awake).
# Optional login: remove the leading # on the next two lines and choose a password.
# $env:APP_USER="jury"
# $env:APP_PASS="change-me"
$env:HF_HOME="D:\hf_cache"
$env:HF_HUB_OFFLINE="1"
$env:HF_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
$env:THRESHOLD="0.25"
$env:BACKEND="hf"
# $env:PROMPT="v2"   # uncomment ONLY if prompt v2 won the benchmark
$env:KB_DIR="corpus"
$env:SHARE="1"
Set-Location "D:\AI\Jury-2 GEN AI"
.\venv\Scripts\Activate.ps1
python app.py