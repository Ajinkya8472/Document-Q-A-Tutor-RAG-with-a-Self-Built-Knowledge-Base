# Document Q&A Tutor (RAG) with a Self-Built Gen-AI Knowledge Base

A tutor that answers questions **only from a set of documents**, shows where each answer came from, and **refuses** when the answer is not in the documents. It runs a small open LLM (Qwen2.5-1.5B) locally on a laptop GPU and can be shared through a temporary public link.

Built for a final-year Generative AI jury (Problem Statement 79: *Textbook Chapter Q&A Tutor Bot*) and extended with a self-built corpus, a generated Q&A dataset and a full benchmark.

## Results

Evaluated on 398 answerable and 40 unanswerable questions over a 97-document knowledge base (4,913 chunks).

| Metric | No LLM (copies sentences) | Qwen2.5-1.5B |
|---|---|---|
| Answer accuracy (key facts present) | 75% | **82%** |
| Correct refusals (question not in documents) | 20% | **92%** |
| Hallucination rate | 80% | **8%** |
| Valid questions wrongly refused | 0% | 7% |
| Right chunk in top 3 (retrieval) | 79% | 79% |
| Citation points to the true source passage | - | 66% |

Speed on an RTX 4050 (6 GB): about 1.9 s per answer on an idle laptop, up to 4.9 s under load. Prompts average 509 tokens (the question plus three retrieved passages).

**Read the limitations below before trusting these numbers.**

## How it works

```
 PDF / TXT  ->  sentence-aware chunks (~110 words, 1-sentence overlap)
                      |
 question  ->  hybrid retrieval: MiniLM embeddings + TF-IDF keywords, top 3
                      |
              Gate 1: best similarity < 0.25  ->  refuse
                      |
              Qwen2.5-1.5B answers from the 3 passages only
                      |
              Gate 2: model replies NOT_IN_CHAPTER  ->  refuse
                      |
              citation attached  +  grounding score
```

## Quick start (Windows PowerShell)

Needs Python 3.10+ (it ran on 3.14 in our setup; 3.11 or 3.12 is the safer choice) and, ideally, an NVIDIA GPU with 6 GB or more. CPU works but is slow.

```powershell
git clone https://github.com/<your-username>/rag-document-tutor.git
cd rag-document-tutor
python -m venv venv
.\venv\Scripts\Activate.ps1

# 1) PyTorch: pick the right command at https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.cuda.is_available())"     # should print True on a GPU machine

# 2) the rest
pip install -r requirements.txt

# 3) build the knowledge base (downloads about 95 Wikipedia articles; needs internet)
python build_corpus.py

# 4) run the app (first run downloads the models, about 3.3 GB)
.\run_demo.ps1
```

Then open the local address printed in the terminal (usually http://127.0.0.1:7860).

Tips:
- **Low disk space on C:?** Run `$env:HF_HOME="D:\hf_cache"` before step 4 so the models are stored elsewhere.
- **Models already downloaded?** Set `$env:HF_OFFLINE="1"` to run with no internet.
- **Add your own documents:** put PDFs or text files in `own_docs/` and run `python build_corpus.py` again; they are copied into `corpus/`. Or upload a file in the app.
- **Linux/macOS:** the `.ps1` scripts only set environment variables. Set `BACKEND=hf`, `KB_DIR=corpus`, `THRESHOLD=0.25`, `HF_MODEL=Qwen/Qwen2.5-1.5B-Instruct` yourself and run `python app.py`.

## Share it with a public link

```powershell
.\run_public.ps1
```

Copy the `https://...gradio.live` link it prints. The link is temporary, changes each run, and only works while your computer stays on. Anyone with the link can use the app: set `APP_USER` and `APP_PASS` (see the script) if you want a login.

## Reproduce the dataset and the benchmark

```powershell
python build_corpus.py                                   # corpus
.\run_build_dataset.ps1                                  # generates data/qa_dataset.json with the local LLM
python review_qa.py data/qa_dataset.json 50              # hand-check 50 pairs
python dataset_card.py corpus qa_dataset_reviewed.json   # corpus and dataset statistics
.\run_benchmark.ps1 corpus qa_dataset_reviewed.json extractive   # no-LLM baseline
.\run_benchmark.ps1 corpus qa_dataset_reviewed.json hf           # with the LLM
python analyze_results.py results_corpus_hf.json qa_dataset_reviewed.json   # failure analysis
```

`review_qa.py` writes `qa_dataset_reviewed.json` in the folder you run it from. Move your final result files into `results/` if you want to keep them in the repo.

## Repository map

| File | What it does |
|---|---|
| `rag_tutor.py` | The whole pipeline: chunking, hybrid retriever, LLM wrapper, refusal gates, grounding score, citation attachment |
| `app.py` | Gradio web app (knowledge base, upload, performance box, query log) |
| `benchmark.py` | Measures model size, tokens, speed, retrieval and answer quality |
| `build_corpus.py` | Downloads Wikipedia articles and copies your own documents into `corpus/` |
| `make_qa.py` | Generates the Q&A dataset from the corpus with the local LLM |
| `review_qa.py` | Command-line tool to hand-verify generated questions |
| `dataset_card.py` | Writes `DATASET_CARD.md` with corpus and dataset statistics |
| `analyze_results.py` | Failure analysis of a benchmark run |
| `grounding_sweep.py` | Tests an optional grounding-score refusal gate (`GROUND_MIN`) |
| `diagnose.py` | Retrieval and refusal-threshold study for a single document |
| `data/` | Hand-written evaluation questions |
| `docs/PRESENTATION_NOTES.md` | Talking points and likely jury questions |

## A learning path (for classmates)

1. Read `rag_tutor.py` from top to bottom. It is one file, in the order the data flows.
2. Run `.\run_demo.ps1` and ask questions. Watch the sources panel and the token and latency box.
3. Try these experiments and compare results with `benchmark.py`:
   - change the chunk size in `chunk_pages` and see how retrieval recall changes;
   - change `THRESHOLD` (0.15, 0.25, 0.35) and watch refusals against wrongly refused questions;
   - set `K=5` for more context per answer;
   - try `PROMPT=v2` (we tested it, and it made refusals worse: see the limitations);
   - swap `HF_MODEL` for `Qwen/Qwen2.5-0.5B-Instruct` and compare speed and accuracy;
   - add your own documents to `own_docs/` and write 20 questions about them.

## Known limitations (please read)

- **Small test set tuned partly on our own data.** The 0.25 threshold was chosen on a separate (LoRA-paper) question set, but the model, prompts and filters were still developed against this project.
- **The Q&A set was generated by the same Qwen model that answers it**, which may favour it. 48 of 50 hand-reviewed pairs were judged good, which means the answer is supported by the passage, not that every question is perfectly worded.
- **Accuracy is a key-phrase match**, which is crude: a correct paraphrase can be marked wrong.
- **Retrieval is the bottleneck:** accuracy is 91% when the right chunk is retrieved and 49% when it is missed.
- **A small model can invent definitions** for made-up names that look related to the documents (3 of 20 in our test).
- **Citations are mostly attached by the system, not written by the model**; 66% point to the true source passage.
- Speed numbers vary with how loaded the laptop is.

## Data, models and licences

- Code: MIT (see `LICENSE`).
- Corpus: Wikipedia text, licensed CC BY-SA 4.0 (attribution: Wikipedia contributors). It is not stored in this repository; `build_corpus.py` downloads it. Any Q&A file that contains passages from it carries the same licence.
- Models are downloaded at run time from Hugging Face: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) and [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Check their licences before reuse.
- PDFs of third-party papers are not included.

## Team

[Add your names and roles here]

## Acknowledgements

Thanks to our subject teacher for guidance on the jury assessment, and to the open-source communities behind Hugging Face Transformers, Sentence-Transformers, scikit-learn and Gradio.
