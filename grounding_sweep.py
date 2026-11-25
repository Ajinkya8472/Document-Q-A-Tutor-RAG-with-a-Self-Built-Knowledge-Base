"""Would a grounding-score gate catch the hallucinations without hurting good answers?
python grounding_sweep.py results_corpus_hf.json"""
import sys, json
rows = json.load(open(sys.argv[1], encoding="utf-8"))
good = [r for r in rows if r["answerable"] and not r["refused"] and r.get("fact_ok")]
bad_answers = [r for r in rows if r["answerable"] and not r["refused"] and not r.get("fact_ok")]
hall = [r for r in rows if not r["answerable"] and not r["refused"]]
print("grounding of the hallucinated answers:", [r["grounding"] for r in hall])
print("median grounding of correct answers:", sorted(r["grounding"] for r in good)[len(good) // 2])
print("\nGROUND_MIN  hallucinations-blocked  correct-answers-lost  wrong-answers-blocked")
for t in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5):
    print(f"  {t:.2f}        {sum(r['grounding'] < t for r in hall)}/{len(hall)}                    "
          f"{sum(r['grounding'] < t for r in good)}/{len(good)}                 {sum(r['grounding'] < t for r in bad_answers)}/{len(bad_answers)}")
print('\nIf a value blocks all/most hallucinations and loses few correct answers, try:  $env:GROUND_MIN="<value>"  then re-run the benchmark.')