"""Generate the self-made Q&A dataset from the corpus with your local LLM.
python make_qa.py corpus qa_dataset.json 400        (BACKEND=hf by default; ~20-30 min for 400 on a GPU)
Each item stores its gold chunk, so retrieval can be scored. Hand-verify a sample afterwards with review_qa.py."""
import sys, os, json, re, math, random, time
from rag_tutor import load_corpus, Generator, words

SYSTEM = "You write exam questions from a passage. Reply with JSON only."
USER = ('Read the passage and write ONE exam-style question that can be answered ONLY from it, plus a short answer.\n'
        'Rules: the question must name its specific subject so it makes sense on its own; never say "the passage", '
        '"the text" or "the article"; the answer is a short phrase or one sentence taken from the passage.\n'
        'Return ONLY this JSON: {"question": "...", "answer": "...", "keywords": ["one or two exact short phrases from the passage that the answer contains"]}\n\nPassage:\n')
BAD = ("the passage", "the text", "this passage", "the article", "this article", "according to the", "the author", "the above", "the given")

NEAR = """What is the Zorvath attention mechanism and who proposed it?|How does the Kellerman-Rao transform improve diffusion models?|What are the main results of the Pendlebury-Nair GAN stability theorem?|Which dataset does the QuillNet language model train on?|What is the Brightwater retrieval benchmark?|How many layers does the Marlowe-7 transformer have?|What is the Tessellate-LoRA fine-tuning method?|Who developed the HyperVolta optimizer?|What does the Fennimore score measure in text generation?|Explain the Orlov-Smith privacy bound for prompt sanitization.|What is the Valence-XL multimodal model?|How does the Draycott algorithm detect hallucinations?|What is the Hallworth tokenizer?|What were the training costs of the Sundial-3 model?|What accuracy did the Iverson GAN reach on ImageNet-12?|Explain Wexler's law of embedding drift.|What is the Mirabel vector index?|Which company released the Aurelia diffusion model?|What is the Calloway-Dunn regularizer?|How does the Thornfield retriever rerank documents?""".split("|")
OFF = """What is the capital of Australia?|Who won the 2018 FIFA World Cup?|How do you make a margherita pizza?|What is the boiling point of ethanol?|Who wrote the novel Pride and Prejudice?|What is the height of Mount Everest?|How many players are in a cricket team?|What is the chemical formula of table salt?|Who painted the Mona Lisa?|Explain how a car engine works.|What are the symptoms of the common cold?|What is the national bird of India?|Which planet is closest to the Sun?|How do vaccines work?|What is the GDP of Brazil?|How do you change a flat tire?|What is the plot of Hamlet?|Who invented the telephone?|What is the tallest building in the world?|How many bones are in the adult human body?""".split("|")

def parse_item(raw, chunk_text):
    """Return a clean item dict or None. Rejects ungrounded / vague / malformed generations."""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m: return None
    try: d = json.loads(m.group(0))
    except Exception: return None
    q, a, kws = str(d.get("question", "")).strip(), str(d.get("answer", "")).strip(), d.get("keywords", [])
    if not isinstance(kws, list) or not q.endswith("?") or not (6 <= len(q.split()) <= 30): return None
    if any(b in q.lower() for b in BAD) or not (1 <= len(a.split()) <= 45): return None
    low = chunk_text.lower()
    kws = [k.strip() for k in kws if isinstance(k, str) and len(k.strip()) >= 3 and k.strip().lower() in low][:2]
    aw = set(words(a))
    if not kws or not aw or len(aw & set(words(chunk_text))) / len(aw) < 0.6: return None
    return {"q": q, "reference_answer": a, "keywords": kws, "min_match": math.ceil(len(kws) / 2)}

def candidates(chunks, seed=42):
    """Round-robin over documents so every document contributes."""
    by = {}
    for c in chunks:
        t = c["text"]
        if len(t.split()) < 70 or sum(ch.isdigit() for ch in t) / len(t) > 0.12 or "ISBN" in t: continue
        by.setdefault(c["doc"], []).append(c)
    rnd = random.Random(seed)
    for v in by.values(): rnd.shuffle(v)
    out, i = [], 0
    while any(by.values()):
        for v in by.values():
            if v: out.append(v.pop())
    return out

if __name__ == "__main__":
    corpus, out_path = sys.argv[1], sys.argv[2]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    chunks = load_corpus(corpus)
    gen = Generator(os.environ.get("BACKEND", "hf"))
    items, seen, tried, t0 = [], set(), 0, time.time()
    part = open(out_path + ".partial.jsonl", "w", encoding="utf-8")
    try:
        for c in candidates(chunks):
            if len(items) >= N: break
            tried += 1
            it = parse_item(gen.chat(SYSTEM, USER + c["text"], 160), c["text"])
            if not it: continue
            key = " ".join(words(it["q"]))
            if key in seen: continue
            seen.add(key)
            it.update(answerable=True, gold_doc=c["doc"], gold_text=c["text"], source="generated", verified=False)
            items.append(it); part.write(json.dumps(it, ensure_ascii=False) + "\n"); part.flush()
            if len(items) % 10 == 0:
                print(f"{len(items)}/{N} accepted from {tried} tries ({time.time()-t0:.0f}s)", flush=True)
    except KeyboardInterrupt:
        print("\nStopped early: saving what was generated so far...")
    unans = [{"q": q, "answerable": False, "kind": k, "source": "hand-written"} for k, qs in (("near-topic fictional", NEAR), ("off-topic", OFF)) for q in qs]
    json.dump(items + unans, open(out_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nSaved {len(items)} answerable + {len(unans)} unanswerable to {out_path}. Yield: {len(items)}/{tried} generations kept.")