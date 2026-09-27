"""[김우진만 실행] 모두가 제출한 ok/ 파일을 합쳐 학습·검증·평가 세트를 만듦.
실행: python ml/data_gen/merge.py
결과: ml/data/train.jsonl (2,000), val.jsonl (200), synth_eval.jsonl (300)
"""
import json
from collections import Counter
from common import HERE, ROOT, TYPES, KO, read_jsonl

SYSTEM = (ROOT / "config" / "system_prompt.txt").read_text(encoding="utf-8").strip()
OUT = ROOT / "ml" / "data"; OUT.mkdir(parents=True, exist_ok=True)

def to_messages(row):
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": row["memo"]},
                         {"role": "assistant", "content": json.dumps(row["gold"], ensure_ascii=False)}]}

def write(name, rows, fmt):
    with open(OUT / name, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(fmt(r), ensure_ascii=False) + "\n")

train = [r for s in ["S1", "S2", "S3", "S4"] for r in read_jsonl(HERE / "ok" / f"{s}.jsonl")]
val = read_jsonl(HERE / "ok" / "W_val.jsonl")
test = read_jsonl(HERE / "ok" / "W_test.jsonl")

# 문체가 세트 사이에 섞이지 않았는지 확인 (평가 문체를 학습에 쓰면 점수가 부풀려짐)
pt, pv, pe = ({r["persona"] for r in x} for x in (train, val, test))
assert not (pt & pv) and not (pt & pe) and not (pv & pe), f"문체가 겹침: {pt & pv} {pt & pe} {pv & pe}"
# 검증·평가 세트와 같은 메모가 학습 세트에 있으면 학습 세트에서 뺌 (평가 점수가 부풀려지지 않게)
held = {r["memo"] for r in val + test}
before = len(train); train = [r for r in train if r["memo"] not in held]
if len(train) < before: print(f"검증·평가와 겹친 메모 {before - len(train)}개를 학습 세트에서 뺌")

write("train.jsonl", train, to_messages)
write("val.jsonl", val, to_messages)
write("synth_eval.jsonl", test, lambda r: r)            # 평가용은 정답(gold)을 그대로 둠

cnt = Counter(e["type"] for r in train for e in r["gold"]["events"])
print(f"train {len(train)} / val {len(val)} / synth_eval {len(test)}")
print("학습 세트 유형별 사건 수:", ", ".join(f"{KO[t]} {cnt[t]}" for t in TYPES))
low = [KO[t] for t in TYPES if cnt[t] < 150]
print("150건 미만 유형:", low or "없음")
print("빈 결과 메모 비율:", round(sum(1 for r in train if not r["gold"]["events"]) / max(len(train), 1), 3))
