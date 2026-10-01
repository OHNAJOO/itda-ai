"""메모 하나를 서비스와 같은 조건(시스템 프롬프트 + 스키마 강제 + temperature 0)으로 넣어 보는 실험용 도구.
실행 (레포 맨 위):
  python ml/eval/try_memo.py                          # 메모를 붙여 넣고 빈 줄에서 Ctrl+D
  python ml/eval/try_memo.py --file 긴메모.txt         # 파일로 넣기
  python ml/eval/try_memo.py --model itda-qwen:q4_k_m --no-schema
출력: 모델 답(JSON 정리), 뽑힌 사건 표, 입력·출력 토큰 수, 걸린 시간, 잘림 여부
"""
import argparse, json, sys, time, urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="itda-gemma4-e4b-q4_k_m")
ap.add_argument("--file", default=None)
ap.add_argument("--no-schema", action="store_true")
ap.add_argument("--num-predict", type=int, default=1024, help="답 최대 길이(토큰). 평가는 384")
ap.add_argument("--num-ctx", type=int, default=4096, help="입력+출력 최대 길이(토큰)")
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--host", default="http://localhost:11434")
args = ap.parse_args()

ROOT = Path.cwd()
system = (ROOT / "config" / "system_prompt.txt").read_text(encoding="utf-8").strip()
if args.file:
    memo = Path(args.file).read_text(encoding="utf-8").strip()
else:
    print("메모를 붙여 넣고, 다 넣었으면 새 줄에서 Ctrl+D", file=sys.stderr)
    memo = sys.stdin.read().strip()

body = {"model": args.model, "stream": False,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": memo}],
        "options": {"temperature": 0, "num_predict": args.num_predict, "num_ctx": args.num_ctx,
                    **({"num_gpu": 0} if args.cpu else {})}}
if not args.no_schema:
    body["format"] = json.loads((ROOT / "config" / "event_schema.json").read_text(encoding="utf-8"))
req = urllib.request.Request(args.host + "/api/chat", data=json.dumps(body).encode("utf-8"),
                             headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=900) as resp:
    res = json.loads(resp.read())
sec = time.time() - t0

raw = res["message"]["content"]
print(f"\n메모 {len(memo)}자 | 입력 {res.get('prompt_eval_count')}토큰, 출력 {res.get('eval_count')}토큰 | {sec:.1f}초 | 종료 이유: {res.get('done_reason')}")
if res.get("done_reason") == "length":
    print("주의: 답이 --num-predict 길이에서 잘림 → JSON이 깨졌을 수 있음. --num-predict를 늘려 다시 실행")
if (res.get("prompt_eval_count") or 0) >= args.num_ctx - 50:
    print("주의: 입력이 --num-ctx에 거의 닿음 → 앞부분이 잘렸을 수 있음. --num-ctx를 늘려 다시 실행")
try:
    events = json.loads(raw)["events"]
except Exception:
    print("JSON으로 읽히지 않음. 원문:\n" + raw); sys.exit()
print(f"\n사건 {len(events)}건")
for e in events:
    ev = e.get("evidence", "")
    mark = "" if ev and ev in memo else "  (근거가 원문에 그대로 없음)"
    print(f"- {e.get('type'):20} {e.get('status'):8} 시간={e.get('time_expr')!s:10} 횟수={e.get('count')}  근거: {ev}{mark}")
