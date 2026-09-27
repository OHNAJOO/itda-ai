"""[방법 B: 채팅] Claude 답을 raw/{slot}.jsonl 끝에 붙임.
쓰는 법: Claude 답 전체를 ml/data_gen/answer.txt에 붙여 넣고 저장 → python ml/data_gen/add_answer.py --slot S1
- {로 시작하는 JSON 줄만 골라 붙임. ``` 표시나 설명 문장은 자동으로 버림
- 이미 붙인 계획 번호와 다른 사람 슬롯 번호는 건너뜀 (같은 답을 두 번 붙여도 괜찮음)
- 다 붙이면 answer.txt를 비움. 다음 묶음 답을 바로 붙여 넣으면 됨
눈 검사에서 틀린 건 지우기: python ml/data_gen/add_answer.py --slot S1 --drop S1-0073,S1-0102
"""
import argparse, json, sys
from pathlib import Path
from common import HERE, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--slot", required=True)
ap.add_argument("--file", default=str(HERE / "answer.txt"))
ap.add_argument("--drop", default="", help="raw에서 지울 계획 번호 (쉼표로 구분)")
args = ap.parse_args()

if args.drop:
    raw_path = HERE / "raw" / f"{args.slot}.jsonl"
    ids = {x.strip() for x in args.drop.split(",") if x.strip()}
    rows = read_jsonl(raw_path)
    keep = [r for r in rows if r.get("id") not in ids]
    with open(raw_path, "w", encoding="utf-8") as out:
        for r in keep: out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[{args.slot}] raw에서 {len(rows) - len(keep)}줄 지움. 검수(check.py)를 다시 돌려 500개가 되는지 확인")
    sys.exit(0)

src = Path(args.file)
if not src.exists() or not src.read_text(encoding="utf-8-sig").strip():
    src.write_text("", encoding="utf-8")
    print(f"{src} 가 비어 있음. Claude 답을 붙여 넣고 저장한 뒤 다시 실행")
    sys.exit(0)

raw_path = HERE / "raw" / f"{args.slot}.jsonl"
raw_path.parent.mkdir(exist_ok=True)
have = {r.get("id") for r in read_jsonl(raw_path)}
prefix = args.slot + "-"
good = dup = bad = other = 0
with open(raw_path, "a", encoding="utf-8") as out:
    for line in open(src, encoding="utf-8-sig"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except Exception:
            bad += 1
            continue
        rid = str(r.get("id", ""))
        if not rid.startswith(prefix):
            other += 1
            continue
        if rid in have:
            dup += 1
            continue
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
        have.add(rid)
        good += 1
src.write_text("", encoding="utf-8")

print(f"[{args.slot}] 새로 붙임 {good}줄 / 이미 있어서 건너뜀 {dup}줄 / 다른 슬롯 번호 {other}줄 / 끊긴 줄 {bad}줄")
if bad:
    print("  끊긴 줄이 있으면 Claude에게 '이어서 계속'이라고 보내고, 받은 답을 다시 answer.txt에 붙여 실행")
if other:
    print(f"  다른 슬롯 번호가 있음. --slot 이 맞는지, 다른 묶음 파일을 붙이지 않았는지 확인")
print(f"  지금까지 {args.slot}에 모인 답: {len(have)}개")
