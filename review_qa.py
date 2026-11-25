"""Hand-verify a sample of the generated questions (this gives you a trusted subset and a data-quality number).
python review_qa.py qa_dataset.json 50"""
import sys, json, random
path, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50
qa = json.load(open(path, encoding="utf-8"))
pool = [x for x in qa if x["answerable"] and not x.get("verified")]
random.seed(1); sample = random.sample(pool, min(n, len(pool)))
good = bad = 0
for i, x in enumerate(sample, 1):
    print(f"\n--- {i}/{len(sample)}  [{x['gold_doc']}]\nPASSAGE: {x['gold_text'][:500]}...\nQ: {x['q']}\nA: {x['reference_answer']}")
    r = input("good (y) / bad (n) / skip (s) / quit (q) > ").strip().lower()
    if r == "q": break
    if r == "y": x["verified"] = True; good += 1
    elif r == "n": x["rejected"] = True; bad += 1
kept = [x for x in qa if not x.get("rejected")]
json.dump(kept, open("qa_dataset_reviewed.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
json.dump([x for x in kept if x.get("verified") or not x["answerable"]], open("qa_verified.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
if good + bad: print(f"\nReviewed {good+bad}: {good} good, {bad} bad -> data quality {100*good/(good+bad):.0f}% good. Saved qa_dataset_reviewed.json and qa_verified.json")