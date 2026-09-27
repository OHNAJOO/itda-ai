"""[방법 B: 채팅] API 키가 없을 때. 계획 20개씩 묶은 프롬프트 파일을 만듦.
실행 예: python ml/data_gen/make_batches.py --slot S1 --personas T01,T05
결과: batches/S1/batch_01.txt ~ batch_30.txt
쓰는 법: 파일 하나를 통째로 복사해 Claude 채팅 창(새 대화)에 붙여 넣음 → 답으로 나온 줄들을
         그대로 복사해 raw/S1.jsonl 끝에 붙여 넣고 저장 → 다음 파일. 500개가 통과하면 멈춰도 됨
"""
import argparse, json
from common import HERE, load_personas, assign_persona, event_line, persona_block, RULES, KO

ap = argparse.ArgumentParser()
ap.add_argument("--slot", required=True)
ap.add_argument("--personas", required=True)
ap.add_argument("--size", type=int, default=20)
args = ap.parse_args()

P = load_personas(); personas = args.personas.split(",")
plans = json.load(open(HERE / "plans" / f"{args.slot}.json", encoding="utf-8"))
out = HERE / "batches" / args.slot; out.mkdir(parents=True, exist_ok=True)
(HERE / "raw").mkdir(exist_ok=True)

for b in range(0, len(plans), args.size):
    chunk = list(enumerate(plans))[b:b + args.size]
    used = sorted({assign_persona(i, personas) for i, _ in chunk})
    parts = [f"너는 치매 환자를 돌보는 가족 보호자다. 아래 계획마다 관찰 메모를 하나씩 쓴다. 계획마다 적힌 문체를 따른다.\n"]
    parts.append("[문체]\n" + "\n\n".join(f"<{pid}>\n{persona_block(P[pid])}" for pid in used))
    for i, plan in chunk:
        ev = "\n".join(event_line(e) for e in plan["events"]) or "(없음. 일상 이야기만 쓴다)"
        ds = "; ".join(plan["distractors"]) or "없음"
        parts.append(f"[계획 {plan['id']}] 문체: {assign_persona(i, personas)}\n사건:\n{ev}\n함께 섞을 문장: {ds}")
    parts.append(RULES)
    parts.append('[답하는 형식] 계획 하나당 JSON 한 줄씩, 계획 순서대로 답한다. 다른 말은 쓰지 않는다. 코드 블록 표시(```)도 쓰지 않는다.\n'
                 '{"id": "계획 번호", "memo": "메모 전체", "evidence": ["사건1의 근거 구절", ...]}\n'
                 '- evidence는 사건 목록과 같은 순서, 같은 개수. 사건이 없으면 []\n'
                 '- 각 근거 구절은 memo 안에 글자 하나 다르지 않게 들어 있어야 한다')
    n = b // args.size + 1
    (out / f"batch_{n:02d}.txt").write_text("\n\n".join(parts), encoding="utf-8")
print(f"{out}에 {n}개 파일을 만듦")
