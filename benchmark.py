"""Full performance report for the jury.
python benchmark.py <pdf | txt | corpus-folder> <questions.json> [extractive|hf|api]
Optional env: HF_MODEL=Qwen/Qwen2.5-1.5B-Instruct  THRESHOLD=0.25  PROMPT=v2  K=3  TAG=kb"""
import sys, os, json, re, numpy as np
from rag_tutor import Tutor, words

path, qfile = sys.argv[1], sys.argv[2]
backend = sys.argv[3] if len(sys.argv) > 3 else "extractive"
t = Tutor(path, backend=backend)
qs = json.load(open(qfile, encoding="utf-8"))
norm = lambda s: re.sub(r"\s+", " ", s.lower().replace("×", "x"))
pages = lambda x: x["page"] if isinstance(x["page"], list) else [x["page"]]
def n_facts(ans, kws):
    a = norm(ans)
    return sum(any(norm(alt) in a for alt in (k if isinstance(k, list) else [k])) for k in kws)
def ref_recall(ans, ref):
    r = set(words(ref)); return len(r & set(words(re.sub(r"\[\d+\]", "", ans)))) / len(r) if r else 0.0

t.ask("warm up")                                   # first LLM call is slow; exclude it
rows = []
for x in qs:
    r = t.ask(x["q"]); st = r["stats"]
    row = dict(q=x["q"], answerable=x["answerable"], refused=r["refused"], gate=r["gate"], answer=r["answer"],
               grounding=r["grounding"], top_score=round(r["sources"][0]["score"], 3),
               top_doc=r["sources"][0].get("doc"), **st)
    if x["answerable"]:
        if "page" in x:
            row["page_hit"] = any(c["page"] in pages(x) for c in r["sources"])
        if "keywords" in x:
            need = x.get("min_match") or len(x["keywords"])
            row["fact_ok"] = (not r["refused"]) and n_facts(r["answer"], x["keywords"]) >= need
        if "reference_answer" in x and not r["refused"]:
            row["ref_recall"] = ref_recall(r["answer"], x["reference_answer"])
        if "gold_text" in x and not r["refused"]:   # is the cited passage really the source passage?
            nums = [int(n) for n in re.findall(r"\[(\d+)\]", r["answer"])]
            row["cited_gold"] = any(1 <= n <= len(r["sources"]) and r["sources"][n - 1]["text"] == x["gold_text"] for n in nums)
        if "gold_text" in x:                        # chunk-level retrieval quality
            top5 = t.retriever.search(x["q"], 5)
            row["gold_rank"] = next((i + 1 for i, c in enumerate(top5) if c["text"] == x["gold_text"]), None)
            row["doc_hit3"] = any(c.get("doc") == x.get("gold_doc") for c in top5[:3])
    rows.append(row)

A = [r for r in rows if r["answerable"]]; U = [r for r in rows if not r["answerable"]]
G = [r for r in rows if r["prompt_tokens"] > 0]
info = t.gen.info(); ctxw = info["context_window"]
emb = getattr(t.retriever, "model", None)
emb_p = round(sum(p.numel() for p in emb.parameters()) / 1e6, 1) if emb is not None else None
pct = lambda n, d: f"{n}/{d} ({100*n/max(d,1):.0f}%)"
out = []
def table(title, rows_):
    out.append(f"\n### {title}\n| Metric | Value |\n|---|---|")
    out.extend(f"| {k} | {v} |" for k, v in rows_)

ndocs = len({c.get("doc") for c in t.chunks})
table("1. LLM used (model & parameters)", [
    ("Backend / model", f"{info['backend']} / {info['model']}"),
    ("Model parameters", f"{info['params_m']} M" if info["params_m"] is not None else "n/a (remote)"),
    ("Model size in memory", f"{info['size_mb']} MB ({info['dtype']}) on {info['device']}" if info["size_mb"] is not None else "n/a"),
    ("Context window (max tokens)", ctxw or "n/a"),
    ("Embedding model", f"all-MiniLM-L6-v2, {emb_p} M params, {t.retriever.emb.shape[1]}-dim" if emb is not None else "TF-IDF (no neural embedder)")])
table("2. Pipeline hyper-parameters", [
    ("Knowledge base", f"{ndocs} document(s), {len(t.chunks)} chunks (~110 words, 1-sentence overlap)"),
    ("Retrieval", f"{t.retriever.kind}" + (" + keyword fusion (alpha=0.5)" if t.retriever.kind == 'dense' and t.retriever.hybrid else "") + f", top-k={t.k}"),
    ("Refusal threshold", t.retriever.threshold),
    ("Prompt version", os.environ.get("PROMPT", "v1")),
    ("Decoding", "greedy (temperature 0), max_new_tokens=200")])
table("3. Token size", [
    ("Prompt tokens per query (avg / max)", f"{np.mean([r['prompt_tokens'] for r in G]):.0f} / {max(r['prompt_tokens'] for r in G)}" if G else "0 (no LLM)"),
    ("Output tokens per answer (avg / max)", f"{np.mean([r['output_tokens'] for r in G]):.0f} / {max(r['output_tokens'] for r in G)}" if G else "0 (no LLM)"),
    ("Context window used (avg)", f"{100*np.mean([r['prompt_tokens'] for r in G])/ctxw:.1f}%" if G and ctxw else "n/a")])
tot = np.array([r["total_s"] for r in rows]); gs = sum(r["generation_s"] for r in G); ot = sum(r["output_tokens"] for r in G)
table("4. Speed", [
    ("Retrieval latency (avg)", f"{1000*np.mean([r['retrieval_s'] for r in rows]):.0f} ms"),
    ("Generation latency (avg, LLM calls only)", f"{np.mean([r['generation_s'] for r in G]):.2f} s" if G else "n/a"),
    ("End-to-end latency (avg / p95)", f"{tot.mean():.2f} s / {np.percentile(tot,95):.2f} s"),
    ("Throughput", f"{ot/gs:.1f} tokens/s" if gs else "n/a")])
gr_ = [r["grounding"] for r in rows if r["grounding"] is not None]
cite = sum(bool(re.search(r"\[\d+\]", r["answer"])) for r in rows if not r["refused"])
Pg = [r for r in A if "page_hit" in r]; Fk = [r for r in A if "fact_ok" in r]; Gd = [r for r in A if "gold_rank" in r]
q5 = [("Valid questions wrongly refused", pct(sum(r["refused"] for r in A), len(A)))]
q = [("Questions (answerable / not in documents)", f"{len(A)} / {len(U)}")]
if Pg: q.append((f"Retrieval hit@{t.k} (right page retrieved)", pct(sum(r["page_hit"] for r in Pg), len(Pg))))
if Gd:
    for k in (1, 3, 5):
        q.append((f"Chunk recall@{k} (gold chunk retrieved)", pct(sum(1 for r in Gd if r["gold_rank"] and r["gold_rank"] <= k), len(Gd))))
    q.append(("MRR@5", f"{np.mean([1/r['gold_rank'] if r['gold_rank'] else 0 for r in Gd]):.3f}"))
    q.append(("Document hit@3 (right document retrieved)", pct(sum(r["doc_hit3"] for r in Gd), len(Gd))))
if Fk: q.append(("Answer accuracy (key facts present)", pct(sum(r["fact_ok"] for r in Fk), len(Fk))))
rr = [r["ref_recall"] for r in A if "ref_recall" in r]
if rr: q.append(("Reference-answer word recall (answered Qs)", f"{np.mean(rr):.2f}"))
q += [("Correct refusals (not-in-documents Qs)", pct(sum(r["refused"] for r in U), len(U))),
      ("Hallucination rate (answered a not-in-documents Q)", pct(sum(not r["refused"] for r in U), len(U)))] + q5 + [
      ("Mean grounding score", f"{np.mean(gr_):.2f}" if gr_ else "n/a"),
      ("Answers with citations (added by the system if the model omits them)", pct(cite, sum(not r["refused"] for r in rows)))]
Cg = [r for r in A if "cited_gold" in r]
if Cg: q.append(("Citation correctness (cited passage is the true source passage)", pct(sum(r["cited_gold"] for r in Cg), len(Cg))))
table("5. Answer quality", q)
report = "\n".join(out)
print(report)
tag = os.environ.get("TAG"); name = f"results_{tag + '_' if tag else ''}{backend}"
open(name + ".md", "w", encoding="utf-8").write(f"# Benchmark: {backend}\n" + report)
json.dump(rows, open(name + ".json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"\nSaved {name}.md and {name}.json (per-question answers, tokens, latency).")