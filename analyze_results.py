"""Failure analysis for a benchmark run (jury material).
python analyze_results.py results_corpus_hf.json qa_dataset_reviewed.json"""
import sys, json, collections
rows = json.load(open(sys.argv[1], encoding="utf-8")); qa = json.load(open(sys.argv[2], encoding="utf-8"))
for r, x in zip(rows, qa): r["kind"] = x.get("kind"); r["doc"] = x.get("gold_doc")
A = [r for r in rows if r["answerable"]]; U = [r for r in rows if not r["answerable"]]
pc = lambda n, d: f"{n}/{d} ({100*n/max(d,1):.0f}%)"

print("== Valid questions wrongly refused ==")
wr = [r for r in A if r["refused"]]
print(" by gate:", dict(collections.Counter(r["gate"] for r in wr)), "(retrieval = low similarity score, llm = model said not in document)")
for r in sorted(wr, key=lambda r: r["top_score"])[:8]: print(f"  - [{r['gate']}, score {r['top_score']}] {r['q'][:90]}")

print("\n== Not-in-document questions the system answered (hallucinations) ==")
for r in U:
    if not r["refused"]: print(f"  - [{r['kind']}] {r['q'][:70]}\n      -> {r['answer'][:120]}")
for k in ("near-topic fictional", "off-topic"):
    g = [r for r in U if r["kind"] == k]
    print(f" refused {k}: {pc(sum(r['refused'] for r in g), len(g))}")

print("\n== Accuracy split by retrieval success ==")
got = [r for r in A if r.get("gold_rank") and r["gold_rank"] <= 3]; miss = [r for r in A if not (r.get("gold_rank") and r["gold_rank"] <= 3)]
print(f" gold chunk in top-3: accuracy {pc(sum(r.get('fact_ok', False) for r in got), len(got))}")
print(f" gold chunk missed:   accuracy {pc(sum(r.get('fact_ok', False) for r in miss), len(miss))}")
ans = [r for r in A if not r["refused"]]
print(f" accuracy among questions it chose to answer: {pc(sum(r.get('fact_ok', False) for r in ans), len(ans))}")

print("\n== Answer behaviour ==")
g = [r for r in rows if r["output_tokens"] > 0]
if g:
    print(f" answers cut off at the 200-token limit: {pc(sum(r['output_tokens'] >= 200 for r in g), len(g))}")
    print(f" answers with a [n] citation: {pc(sum('[' in r['answer'] and ']' in r['answer'] for r in g if not r['refused']), sum(not r['refused'] for r in g))}")

print("\n== Weakest documents (>=4 questions) ==")
by = collections.defaultdict(list)
for r in A: by[r["doc"]].append(r.get("fact_ok", False))
for d, v in sorted(((d, v) for d, v in by.items() if len(v) >= 4), key=lambda t: sum(t[1]) / len(t[1]))[:5]:
    print(f"  {d}: {sum(v)}/{len(v)}")