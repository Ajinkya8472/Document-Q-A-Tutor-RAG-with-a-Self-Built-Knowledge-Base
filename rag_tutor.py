"""Textbook Chapter Q&A Tutor (RAG) - core pipeline.
Member A: load_pages / chunk_pages   Member B: Retriever
Member C: Generator                  Member D: Tutor.ask + grounding (see eval.py, app.py)
"""
import os, re, time
import numpy as np

STOP = set("a an the of to in on and or is are was were be been it its this that for with as by at from "
           "which what who how why when can does do did not no than then so such these those their they".split())

# ---------------- A. Ingestion ----------------
def load_pages(path):
    """Return [(page_no, text)] from a PDF or .txt file."""
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(path).pages)]
    with open(path, encoding="utf-8") as f:
        return [(1, f.read())]

def _sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    return [x for x in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\u2022\[(])", text) if x]

def chunk_pages(pages, size=110, overlap_sents=1):
    """Sentence-aware chunks (~size words) with a 1-sentence overlap; each keeps its page number."""
    chunks = []
    for pg, text in pages:
        sents = []
        for x in _sentences(text):
            w = x.split()
            if len(w) > size:                       # very long runs (e.g. flattened tables)
                sents += [" ".join(w[i:i + size]) for i in range(0, len(w), size)]
            else:
                sents.append(x)
        cur, n = [], 0
        for x in sents:
            L = len(x.split())
            if cur and n + L > size:
                chunks.append({"id": len(chunks), "page": pg, "text": " ".join(cur)})
                cur = cur[-overlap_sents:] if overlap_sents else []
                n = sum(len(y.split()) for y in cur)
            cur.append(x); n += L
        if cur:
            chunks.append({"id": len(chunks), "page": pg, "text": " ".join(cur)})
    return chunks

def load_corpus(folder):
    """Load every .pdf/.txt/.md in a folder into one chunk list; each chunk remembers its document."""
    chunks = []
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith((".pdf", ".txt", ".md")):
            continue
        try:
            pages = load_pages(os.path.join(folder, fn))
        except Exception as e:
            print("skipped", fn, "->", e)
            continue
        for c in chunk_pages(pages):
            c["doc"] = fn
            chunks.append(c)
    for i, c in enumerate(chunks):
        c["id"] = i
    return chunks

# ---------------- B. Retrieval ----------------
class Retriever:
    """Dense retrieval + keyword (TF-IDF) fusion. Falls back to TF-IDF only if embeddings are unavailable.
    Ranking uses fused = dense + alpha*tfidf; the reported `score` is the dense cosine (used for refusal)."""
    def __init__(self, chunks, model="sentence-transformers/all-MiniLM-L6-v2", force_tfidf=False,
                 hybrid=True, alpha=0.5):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.chunks, self.hybrid, self.alpha = chunks, hybrid, alpha
        texts = [c["text"] for c in chunks]
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.tf = self.vec.fit_transform(texts)
        self.kind, self.threshold = "tfidf", 0.08
        if not force_tfidf:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model)
                self.emb = self.model.encode(texts, normalize_embeddings=True)
                self.kind, self.threshold = "dense", 0.30      # tune with diagnose.py
            except Exception:
                pass

    def search(self, query, k=3):
        t = (self.tf @ self.vec.transform([query]).T).toarray().ravel()
        if self.kind == "dense":
            d = self.emb @ self.model.encode([query], normalize_embeddings=True)[0]
            rank = d + self.alpha * t if self.hybrid else d
            score = d
        else:
            rank = score = t
        top = np.argsort(-rank)[:k]
        return [dict(self.chunks[i], score=float(score[i])) for i in top]

# ---------------- C. Generation ----------------
SYSTEM_V1 = ("You are a patient tutor. Answer ONLY using the numbered context passages. "
             "Cite passages like [1] after each claim. If the context does not contain the answer, "
             "reply exactly: NOT_IN_CHAPTER. Keep the answer under 120 words.")
SYSTEM_V2 = ("You are a precise tutor. Use ONLY the numbered context passages. Answer in 1-3 short sentences. "
             "End EVERY sentence with the passage number in square brackets, like [1] or [2]. "
             "Copy numbers, units and names exactly as written in the passages. "
             "If the passages do not contain the answer, reply exactly: NOT_IN_CHAPTER.")
SYSTEM = SYSTEM_V2 if os.environ.get("PROMPT") == "v2" else SYSTEM_V1   # A/B test with $env:PROMPT="v2"

def build_prompt(question, ctx):
    block = "\n\n".join(f"[{i+1}] (page {c['page']}) {c['text']}" for i, c in enumerate(ctx))
    return f"Context:\n{block}\n\nQuestion: {question}\nAnswer:"

class Generator:
    """backend: 'hf' (local open LLM), 'api' (OpenAI-compatible endpoint), 'extractive' (no LLM).
    Every call records token counts and latency in self.last; info() describes the model."""
    def __init__(self, backend="extractive", hf_model=None):
        self.backend = backend
        self.hf_model = hf_model or os.environ.get("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        self.last = dict(prompt_tokens=0, output_tokens=0, latency_s=0.0)
        if backend == "hf":
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tok = AutoTokenizer.from_pretrained(self.hf_model)
            dt = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(self.hf_model, dtype=dt).to(self.device).eval()

    def info(self):
        if self.backend == "hf":
            ps = list(self.model.parameters())
            return dict(backend="hf", model=self.hf_model, params_m=round(sum(p.numel() for p in ps) / 1e6, 1),
                        context_window=getattr(self.model.config, "max_position_embeddings", None),
                        dtype=str(ps[0].dtype).replace("torch.", ""), device=self.device,
                        size_mb=round(sum(p.numel() * p.element_size() for p in ps) / 1e6))
        if self.backend == "api":
            return dict(backend="api", model=os.environ.get("LLM_MODEL", "?"), params_m=None,
                        context_window=None, dtype="n/a", device="remote", size_mb=None)
        return dict(backend="extractive", model="rule-based sentence ranker (no LLM)", params_m=0,
                    context_window=None, dtype="n/a", device="cpu", size_mb=0)

    def chat(self, system, user, max_new_tokens=200):
        """Plain chat call (used to generate the Q&A dataset)."""
        if self.backend == "hf":
            import torch
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = self.tok(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if self.backend == "api":
            import requests
            r = requests.post(os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + os.environ["LLM_API_KEY"]},
                json={"model": os.environ["LLM_MODEL"], "temperature": 0,
                      "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                timeout=60).json()
            return r["choices"][0]["message"]["content"].strip()
        raise RuntimeError("chat() needs the hf or api backend")

    def answer(self, question, ctx):
        prompt = build_prompt(question, ctx)
        t0 = time.perf_counter()
        if self.backend == "hf":
            import torch
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = self.tok(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=200, do_sample=False)
            new = out[0][enc["input_ids"].shape[1]:]
            ans = self.tok.decode(new, skip_special_tokens=True).strip()
            self.last = dict(prompt_tokens=int(enc["input_ids"].shape[1]), output_tokens=int(len(new)),
                             latency_s=time.perf_counter() - t0)
            return ans
        if self.backend == "api":
            import requests   # set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
            r = requests.post(os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + os.environ["LLM_API_KEY"]},
                json={"model": os.environ["LLM_MODEL"], "temperature": 0,
                      "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]},
                timeout=60).json()
            u = r.get("usage", {})
            self.last = dict(prompt_tokens=u.get("prompt_tokens", 0), output_tokens=u.get("completion_tokens", 0),
                             latency_s=time.perf_counter() - t0)
            return r["choices"][0]["message"]["content"].strip()
        ans = self._extractive(question, ctx)
        self.last = dict(prompt_tokens=0, output_tokens=0, latency_s=time.perf_counter() - t0)
        return ans

    @staticmethod
    def _extractive(question, ctx):
        """Baseline: return the passage sentences that best overlap the question."""
        qw = set(words(question))
        sents = []
        for i, c in enumerate(ctx):
            for s in re.split(r"(?<=[.!?])\s+", c["text"]):
                sents.append((len(qw & set(words(s))), s, i + 1))
        best = sorted(sents, key=lambda t: -t[0])[:2]
        return " ".join(f"{s} [{n}]" for _, s, n in best)

# ---------------- D. Tutor + grounding ----------------
def words(t):
    return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP]

def grounding_score(answer, ctx):
    """Fraction of the answer's content words that appear in the retrieved context (0-1)."""
    a = set(words(re.sub(r"\[\d+\]", "", answer)))
    c = set(words(" ".join(x["text"] for x in ctx)))
    return len(a & c) / len(a) if a else 0.0

def ensure_citation(answer, ctx):
    """Small models often ignore 'cite [n]'. If no citation is present, attach the retrieved passage
    whose words best overlap the answer (post-hoc attribution). Citations are added by the system, not the model."""
    if re.search(r"\[\d+\]", answer):
        return answer
    aw = set(words(answer))
    if not aw or not ctx:
        return answer
    sc = [len(aw & set(words(c["text"]))) / len(aw) for c in ctx]
    best = max(range(len(sc)), key=sc.__getitem__)
    return f"{answer} [{best + 1}]" if sc[best] > 0 else answer

REFUSAL = "I couldn't find that in the documents I have. Try rephrasing, or ask about a topic they cover."

class Tutor:
    def __init__(self, path=None, backend="extractive", force_tfidf=False, k=3, text=None, gen=None):
        if text:                                                   # pasted text
            self.chunks = chunk_pages([(1, text)])
            for c in self.chunks: c["doc"] = "pasted text"
        elif path and os.path.isdir(path):                         # a whole knowledge base (folder)
            self.chunks = load_corpus(path)
        else:                                                      # one PDF/TXT file
            self.chunks = chunk_pages(load_pages(path))
            for c in self.chunks: c["doc"] = os.path.basename(path)
        if not self.chunks:
            raise ValueError("No readable text found (scanned PDFs need OCR first).")
        self.retriever = Retriever(self.chunks, force_tfidf=force_tfidf)
        if os.environ.get("THRESHOLD"):
            self.retriever.threshold = float(os.environ["THRESHOLD"])
        self.gen = gen or Generator(backend)                       # pass gen= to share one LLM across tutors
        self.k = int(os.environ.get("K", k))

    def ask(self, question):
        t0 = time.perf_counter()
        ctx = self.retriever.search(question, self.k)
        stats = dict(retrieval_s=time.perf_counter() - t0, prompt_tokens=0, output_tokens=0, generation_s=0.0)
        def done(**kw):
            stats["total_s"] = time.perf_counter() - t0
            return dict(sources=ctx, stats=stats, **kw)
        if ctx[0]["score"] < self.retriever.threshold:               # gate 1: retrieval confidence
            return done(answer=REFUSAL, refused=True, grounding=None, gate="retrieval")
        ans = self.gen.answer(question, ctx)
        g = self.gen.last
        stats.update(prompt_tokens=g["prompt_tokens"], output_tokens=g["output_tokens"], generation_s=g["latency_s"])
        if "NOT_IN_CHAPTER" in ans:                                  # gate 2: LLM says unsupported
            return done(answer=REFUSAL, refused=True, grounding=None, gate="llm")
        ans = ensure_citation(ans, ctx)
        g = grounding_score(ans, ctx)
        if float(os.environ.get("GROUND_MIN", 0)) and g < float(os.environ["GROUND_MIN"]):   # optional gate 3 (off by default)
            return done(answer=REFUSAL, refused=True, grounding=None, gate="grounding")
        return done(answer=ans, refused=False, grounding=round(g, 2), gate=None)