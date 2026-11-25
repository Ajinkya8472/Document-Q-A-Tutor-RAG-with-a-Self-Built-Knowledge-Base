"""Document Q&A Tutor (RAG).  python app.py
Starts with the knowledge base in ./corpus (KB_DIR) loaded; users may also upload one PDF/TXT or paste text.
Env: BACKEND=hf|api|extractive  KB_DIR=corpus  SHARE=1 (public link)  APP_USER / APP_PASS (optional login)  FORCE_TFIDF=1"""
import os, json, time, gradio as gr
from rag_tutor import Tutor, Generator

BACKEND = os.environ.get("BACKEND", "extractive")
FORCE = os.environ.get("FORCE_TFIDF") == "1"
KB_DIR = os.environ.get("KB_DIR", "corpus")
GEN = Generator(BACKEND)                       # one LLM shared by every tutor (loaded once)

KB, kb_info = None, "No knowledge base folder found: upload a file or paste text."
if os.path.isdir(KB_DIR) and any(f.lower().endswith((".pdf", ".txt", ".md")) for f in os.listdir(KB_DIR)):
    KB = Tutor(KB_DIR, backend=BACKEND, force_tfidf=FORCE, gen=GEN)
    kb_info = (f"Knowledge base: {len({c['doc'] for c in KB.chunks})} documents, {len(KB.chunks)} chunks "
               f"({KB.retriever.kind} retriever, {BACKEND} generator)")
    if BACKEND != "extractive": KB.ask("warm up")
os.makedirs("logs", exist_ok=True)

def build(file, pasted):
    try:
        if pasted and pasted.strip():
            t = Tutor(text=pasted, backend=BACKEND, force_tfidf=FORCE, gen=GEN)
        elif file is not None:
            t = Tutor(getattr(file, "name", file), backend=BACKEND, force_tfidf=FORCE, gen=GEN)
        else:
            return None, "Upload a PDF/TXT or paste text first (or click 'Use knowledge base')."
    except Exception as e:
        return None, f"Could not read the text: {e}"
    return t, f"Using YOUR document: {len(t.chunks)} chunks indexed."

def use_kb():
    return None, kb_info

def ask(tutor, q):
    t = tutor or KB
    if t is None or not q.strip():
        return "Load a document first, or ask a question.", "n/a", "", ""
    r = t.ask(q)
    src = "\n\n".join(
        f"[{i+1}] {c.get('doc','')}{' p'+str(c['page']) if str(c.get('doc','')).lower().endswith('.pdf') else ''} "
        f"(score {c['score']:.2f}): {c['text'][:300]}..." for i, c in enumerate(r["sources"]))
    g = "n/a (refused)" if r["grounding"] is None else f"{r['grounding']:.0%}"
    st = r["stats"]
    stats = (f"retrieval {st['retrieval_s']*1000:.0f} ms | generation {st['generation_s']:.1f} s | "
             f"prompt tokens {st['prompt_tokens']} | output tokens {st['output_tokens']}")
    with open("logs/queries.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), q=q, refused=r["refused"],
                                total_s=round(st["total_s"], 2), docs=[c.get("doc") for c in r["sources"]]),
                           ensure_ascii=False) + "\n")
    return r["answer"], g, src, stats

with gr.Blocks(title="Document Q&A Tutor (RAG)") as demo:
    gr.Markdown("# Document Q&A Tutor (RAG)\nAsk anything about the Gen-AI knowledge base, or load your own document. "
                "Answers come only from the documents, with sources; if the answer is not there, the tutor refuses.")
    state = gr.State(None)
    with gr.Row():
        with gr.Column():
            status = gr.Textbox(label="Status", value=kb_info, interactive=False)
            with gr.Accordion("Use your own document instead", open=False):
                file = gr.File(label="Upload PDF or TXT", file_types=[".pdf", ".txt"])
                pasted = gr.Textbox(label="...or paste text here", lines=6)
                load = gr.Button("Load my document", variant="primary")
                back = gr.Button("Back to knowledge base")
        with gr.Column():
            q = gr.Textbox(label="Ask a question")
            go = gr.Button("Ask", variant="primary")
            a = gr.Textbox(label="Answer")
            g = gr.Textbox(label="Grounding score")
            st = gr.Textbox(label="Performance (latency / tokens)")
            s = gr.Textbox(label="Retrieved sources", lines=8)
    load.click(build, [file, pasted], [state, status])
    back.click(use_kb, None, [state, status])
    go.click(ask, [state, q], [a, g, s, st])
    q.submit(ask, [state, q], [a, g, s, st])

auth = (os.environ["APP_USER"], os.environ["APP_PASS"]) if os.environ.get("APP_USER") and os.environ.get("APP_PASS") else None
demo.queue().launch(share=os.environ.get("SHARE") == "1", auth=auth)