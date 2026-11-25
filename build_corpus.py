"""Build the knowledge base folder.   python build_corpus.py
1) Downloads Gen-AI topic articles from Wikipedia (CC BY-SA) into corpus/wiki_*.txt
2) Copies your own documents from own_docs/ (PDF/TXT/DOCX-converted) into corpus/
Needs internet. Re-running skips files already downloaded."""
import os, re, time, shutil, requests

TITLES = """Generative adversarial network|Transformer (deep learning)|Large language model|Retrieval-augmented generation|GPT-3|GPT-4|BERT (language model)|Attention (machine learning)|Word embedding|Autoencoder|Variational autoencoder|Diffusion model|Stable Diffusion|Prompt engineering|Fine-tuning (deep learning)|LoRA (machine learning)|Reinforcement learning from human feedback|Convolutional neural network|Recurrent neural network|Long short-term memory|Backpropagation|Gradient descent|Overfitting|Transfer learning|Self-supervised learning|Unsupervised learning|Supervised learning|Neural network (machine learning)|Deep learning|Machine learning|Artificial intelligence|Generative artificial intelligence|Text-to-image model|DALL-E|Midjourney|ChatGPT|Hallucination (artificial intelligence)|Byte pair encoding|Vector database|Cosine similarity|Semantic search|Information retrieval|tf–idf|Okapi BM25|Bayesian inference|Probability distribution|Normal distribution|Latent space|StyleGAN|Deepfake|Speech synthesis|Whisper (speech recognition system)|Neural machine translation|Natural language processing|Sentiment analysis|Named-entity recognition|Question answering|Automatic summarization|Multimodal learning|Foundation model|Prompt injection|AI alignment|Algorithmic bias|Differential privacy|Federated learning|Knowledge distillation|Mixture of experts|Zero-shot learning|Few-shot learning|Sentence embedding|Language model|Word2vec|Generative pre-trained transformer|Text-to-speech|Image segmentation|Object detection|Computer vision|Optical character recognition|Speech recognition|Recommender system|Cluster analysis|K-means clustering|Principal component analysis|Regularization (mathematics)|Activation function|Loss function|Tokenization (lexical analysis)|Claude (language model)|LLaMA|Gemini (language model)|Mistral AI|Hugging Face|OpenAI|Anthropic|Turing test""".split("|")

CUT = re.compile(r"\n==+ *(See also|References|External links|Further reading|Notes|Bibliography|Sources|Footnotes) *==+.*", re.S | re.I)
HDR = re.compile(r"\n=+ *(.+?) *=+ *\n")
API = "https://en.wikipedia.org/w/api.php"
HEAD = {"User-Agent": "StudentRAGProject/1.0 (college jury project; contact: student)"}

def fetch(title, retries=3):
    for attempt in range(retries):
        r = requests.get(API, headers=HEAD, timeout=30, params=dict(
            action="query", prop="extracts", explaintext=1, redirects=1, format="json", titles=title))
        if not r.text.strip():
            print(f"  empty response for {title}, retry {attempt+1}/{retries}...")
            time.sleep(3)
            continue
        page = next(iter(r.json()["query"]["pages"].values()))
        return page.get("title", title), page.get("extract", "") or ""
    raise RuntimeError(f"Empty response after {retries} retries")

os.makedirs("corpus", exist_ok=True)
ok = skipped = 0; words = 0
for t in TITLES:
    fn = "wiki_" + re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_") + ".txt"
    path = os.path.join("corpus", fn)
    if os.path.exists(path):
        ok += 1; continue
    try:
        title, text = fetch(t)
    except Exception as e:
        print("FAILED", t, e); skipped += 1; continue
    text = CUT.sub("", "\n" + text)
    text = HDR.sub(lambda m: "\n" + m.group(1) + ".\n", text).strip()
    if len(text) < 500:
        print("too short / missing:", t); skipped += 1; continue
    with open(path, "w", encoding="utf-8") as f:
        f.write(title + "\n\n" + text)
    ok += 1; words += len(text.split()); print("saved", fn, len(text.split()), "words")
    time.sleep(1.5)

if os.path.isdir("own_docs"):
    for fn in os.listdir("own_docs"):
        if fn.lower().endswith((".pdf", ".txt", ".md")):
            shutil.copy(os.path.join("own_docs", fn), os.path.join("corpus", fn)); print("copied own doc:", fn)
print(f"\nWikipedia articles: {ok} ok, {skipped} skipped. Folder 'corpus' now holds {len(os.listdir('corpus'))} files.")