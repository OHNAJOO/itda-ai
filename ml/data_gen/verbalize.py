"""[방법 A: API] 계획을 대형 LLM에 보내 메모로 풀어 씀. 중간에 끊겨도 다시 실행하면 이어서 함.
실행 예: python ml/data_gen/verbalize.py --slot S1 --personas T01,T05 --limit 5   (시험 5개)
         python ml/data_gen/verbalize.py --slot S1 --personas T01,T05             (전체)
         python ml/data_gen/verbalize.py --slot S1 --personas T01,T05 --redo      (떨어진 것만 다시)
결과: raw/S1.jsonl (한 줄에 메모 하나)
준비: pip install anthropic pyyaml, 환경 변수 ANTHROPIC_API_KEY 설정
"""
import argparse, json, os, sys, time
from common import HERE, LLM_MODEL, load_personas, assign_persona, build_prompt, parse_json_obj, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--slot", required=True)            # S1~S4, W_val, W_test
ap.add_argument("--personas", required=True)        # 쉼표로 구분. 예: T01,T05
ap.add_argument("--limit", type=int, default=0)     # 시험 삼아 몇 개만 돌릴 때
ap.add_argument("--redo", action="store_true")      # check.py에서 떨어진 것만 지우고 다시 문장화
args = ap.parse_args()

import anthropic
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
P = load_personas()
personas = args.personas.split(",")
for p in personas:
    if p not in P: sys.exit(f"없는 문체: {p}")

plans = json.load(open(HERE / "plans" / f"{args.slot}.json", encoding="utf-8"))
out_path = HERE / "raw" / f"{args.slot}.jsonl"; out_path.parent.mkdir(exist_ok=True)
rows = read_jsonl(out_path)
if args.redo:
    fp = HERE / "ok" / f"{args.slot}_failed.txt"
    bad = set(fp.read_text(encoding="utf-8").split()) if fp.exists() else set()
    rows = [r for r in rows if r["id"] not in bad]
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"떨어진 {len(bad)}개를 지우고 다시 문장화함")
done = {r["id"] for r in rows}
todo = [(i, p) for i, p in enumerate(plans) if p["id"] not in done]
if args.limit: todo = todo[:args.limit]
print(f"{args.slot}: 전체 {len(plans)}, 이미 {len(done)}, 이번에 {len(todo)}")

with open(out_path, "a", encoding="utf-8") as f:
    for k, (i, plan) in enumerate(todo, 1):
        pid = assign_persona(i, personas)
        prompt = build_prompt(plan, P[pid])
        for attempt in range(3):
            try:
                msg = client.messages.create(model=LLM_MODEL, max_tokens=600, temperature=1.0,
                                             messages=[{"role": "user", "content": prompt}])
                obj = parse_json_obj(msg.content[0].text)
                row = {"id": plan["id"], "persona": pid, "memo": obj["memo"], "evidence": obj["evidence"]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
                break
            except Exception as e:
                print(f"  {plan['id']} 실패 {attempt + 1}회: {e}")
                time.sleep(3)
        if k % 50 == 0:
            print(f"  {k}/{len(todo)} 완료")
print("끝. 다음 단계: check.py")
