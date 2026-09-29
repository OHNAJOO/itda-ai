"""평가 질문 3 (기존 능력은 유지되는가): 베이스와 파인튜닝 모델에 일반 질문을 똑같이 넣고 정답률을 비교함.
시스템 프롬프트 없이, 같은 생성 설정(temperature 0)으로 부름. 채점은 전부 코드로 함 (사람 판단 없음).

세트
  A. 일반 질문 30개 (ml/eval/data/regression_qa.jsonl): 상식 8, 계산 6, 높임말·어휘 6, 짧은 독해 5, 건강·돌봄 상식 5
     채점: 답에 허용 정답 중 하나가 들어 있으면 정답
  B. KMMLU 50문항 (공개 한국어 벤치마크, 5과목(한국사·심리·보건·교육·사회복지) × 10문항, 시드 고정): 4지선다. 답에서 처음 나온 1~4 숫자로 채점
보조 지표
  json_contamination: 일반 질문에 {"events": ...} 같은 추출 JSON으로 답한 비율 (추출 전용으로 특화된 정도)

준비 (한 번): uv pip install datasets
실행 (레포 맨 위):
  python ml/eval/regression_eval.py --models gemma4:e4b-it-q4_K_M itda-gemma4-e4b:q4_k_m
결과: ml/eval/results/regression_<모델>.json, ml/eval/results/regression_summary.md
판단 기준 (강사님 자료 예시): 정답률 하락이 3%p 이내면 '기존 능력 유지'
"""
import argparse, json, random, re, time, urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--kmmlu-subjects", nargs="+",
                default=["Korean-History", "Psychology", "Health", "Education", "Social-Welfare"])
ap.add_argument("--kmmlu-per-subject", type=int, default=10)
ap.add_argument("--host", default="http://localhost:11434")
args = ap.parse_args()

ROOT = Path.cwd()
DATA = ROOT / "ml" / "eval" / "data"; DATA.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "ml" / "eval" / "results"; OUT.mkdir(parents=True, exist_ok=True)

# ---------- A. 일반 질문 30개 ----------
QA = [
 # 상식
 ("상식", "대한민국의 수도는 어디인가요?", ["서울"]),
 ("상식", "한글을 만든 조선의 왕은 누구인가요?", ["세종"]),
 ("상식", "1기압에서 물이 끓는 온도는 섭씨 몇 도인가요?", ["100"]),
 ("상식", "윤년이 아닌 해의 1년은 며칠인가요?", ["365"]),
 ("상식", "태양계에서 가장 큰 행성은 무엇인가요?", ["목성"]),
 ("상식", "우리나라 광복절은 몇 월 며칠인가요?", ["8월 15일", "8월15일", "8.15"]),
 ("상식", "무지개 색에서 빨강 바로 다음 색은 무엇인가요?", ["주황"]),
 ("상식", "우리나라에서 구급차를 부를 때 누르는 전화번호는 무엇인가요?", ["119"]),
 # 계산
 ("계산", "17 더하기 25는 얼마인가요?", ["42"]),
 ("계산", "12 곱하기 12는 얼마인가요?", ["144"]),
 ("계산", "사과 3개가 1,500원이면 사과 7개는 얼마인가요?", ["3500", "3,500", "삼천오백"]),
 ("계산", "오전 9시에 약을 먹고 8시간 뒤에 다시 먹어야 합니다. 몇 시에 먹어야 하나요?", ["오후 5시", "17시", "오후 다섯 시", "오후 5 시"]),
 ("계산", "100에서 37을 빼면 얼마인가요?", ["63"]),
 ("계산", "2를 10번 곱하면 얼마인가요?", ["1024", "1,024"]),
 # 높임말·어휘
 ("어휘", "'밥'의 높임말은 무엇인가요?", ["진지"]),
 ("어휘", "'나이'의 높임말은 무엇인가요?", ["연세", "춘추"]),
 ("어휘", "'먹다'의 높임말은 무엇인가요?", ["드시다", "잡수시다", "잡수다"]),
 ("어휘", "'자다'의 높임말은 무엇인가요?", ["주무시다"]),
 ("어휘", "'빠르다'의 반대말은 무엇인가요?", ["느리다"]),
 ("어휘", "'옷을 입다'에서 '입다'의 반대말은 무엇인가요?", ["벗다"]),
 # 짧은 독해
 ("독해", "다음 글을 읽고 답하세요. '민수는 월요일에 병원에 갔고, 수요일에 약국에 갔다.' 민수가 약국에 간 요일은?", ["수요일"]),
 ("독해", "다음 글을 읽고 답하세요. '할머니는 아침에 산책을 하시고 점심을 드신 뒤 낮잠을 주무신다.' 할머니는 언제 산책을 하시나요?", ["아침"]),
 ("독해", "다음 글을 읽고 답하세요. '영희는 사과를 5개 샀고 그중 2개를 먹었다.' 남은 사과는 몇 개인가요?", ["3개", "세 개", "3 개"]),
 ("독해", "다음 글을 읽고 답하세요. '기차는 10시에 출발해서 12시 30분에 도착했다.' 걸린 시간은?", ["2시간 30분", "2시간30분", "두 시간 30분", "150분", "2시간 반", "두 시간 반"]),
 ("독해", "다음 글을 읽고 답하세요. '철수는 영희보다 키가 크고, 영희는 민지보다 키가 크다.' 가장 키가 작은 사람은?", ["민지"]),
 # 건강·돌봄 상식
 ("돌봄", "치매의 원인 질환 가운데 가장 흔한 것은 무엇인가요?", ["알츠하이머"]),
 ("돌봄", "혈압을 재는 기구의 이름은 무엇인가요?", ["혈압계"]),
 ("돌봄", "탈수를 막으려면 무엇을 자주 마시는 것이 좋나요? 한 단어로 답하세요.", ["물"]),
 ("돌봄", "노인의 낙상을 막기 위해 욕실 바닥에 까는 것은 무엇인가요?", ["미끄럼", "논슬립", "미끄러짐 방지"]),
 ("돌봄", "당뇨병 환자가 주로 관리해야 하는 혈액 속 수치는 무엇인가요?", ["혈당"]),
]
qa_path = DATA / "regression_qa.jsonl"
with open(qa_path, "w", encoding="utf-8") as f:
    for i, (cat, q, ans) in enumerate(QA):
        f.write(json.dumps({"id": f"q{i + 1:02d}", "category": cat, "question": q, "answers": ans}, ensure_ascii=False) + "\n")

# ---------- B. KMMLU 50문항 (한 번 받아서 고정 저장) ----------
km_path = DATA / "kmmlu_50.jsonl"
if not km_path.exists():
    from datasets import load_dataset
    rng = random.Random(20260929); rows = []
    for subj in args.kmmlu_subjects:
        ds = load_dataset("HAERAE-HUB/KMMLU", subj, split="test")
        for j in rng.sample(range(len(ds)), args.kmmlu_per_subject):
            r = ds[j]
            rows.append({"id": f"{subj}-{j}", "subject": subj, "question": r["question"],
                         "options": [r["A"], r["B"], r["C"], r["D"]], "answer": int(r["answer"])})
    with open(km_path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("KMMLU 저장:", km_path, len(rows), "문항")
KM = [json.loads(l) for l in open(km_path, encoding="utf-8")]
QA_ROWS = [json.loads(l) for l in open(qa_path, encoding="utf-8")]

# ---------- 호출·채점 ----------
def chat(model, text, num_predict):
    body = {"model": model, "stream": False, "keep_alive": "30m",
            "messages": [{"role": "user", "content": text}], "think": False,
            "options": {"temperature": 0, "num_predict": num_predict}}
    req = urllib.request.Request(args.host + "/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            m = json.loads(resp.read())["message"]
            return m.get("content") or m.get("thinking") or ""
    except Exception as e:
        return f"<<ERROR>> {e}"

def norm(s): return re.sub(r"\s+", " ", s).strip()
def is_json_like(s): return '"events"' in s or s.strip().startswith("{")
def first_choice(s):
    m = re.search(r"[1-4]", s)
    if m: return int(m.group())
    m = re.search(r"\b([A-D])\b", s)
    return "ABCD".index(m.group(1)) + 1 if m else None

summary = []
for model in args.models:
    t0 = time.time(); recs = []
    chat(model, "안녕하세요", 8)                            # 모델 올려 두기
    for r in QA_ROWS:
        out = chat(model, f"다음 질문에 짧게 답하세요.\n질문: {r['question']}", 128)
        ok = any(a in norm(out) for a in r["answers"])
        recs.append({"set": "qa", "id": r["id"], "category": r["category"], "question": r["question"],
                     "answers": r["answers"], "output": out, "correct": ok, "json_like": is_json_like(out)})
    for r in KM:
        opts = "\n".join(f"{k + 1}. {o}" for k, o in enumerate(r["options"]))
        out = chat(model, f"다음 문제의 정답 번호(1~4)만 숫자 하나로 답하세요.\n\n{r['question']}\n{opts}\n정답:", 16)
        pred = first_choice(out)
        recs.append({"set": "kmmlu", "id": r["id"], "category": r["subject"], "answer": r["answer"],
                     "output": out, "pred": pred, "correct": pred == r["answer"], "json_like": is_json_like(out)})
    # 모델 내리기
    try:
        urllib.request.urlopen(urllib.request.Request(args.host + "/api/generate",
            data=json.dumps({"model": model, "keep_alive": 0}).encode(), headers={"Content-Type": "application/json"}), timeout=60)
    except Exception: pass

    def acc(sel): return round(sum(x["correct"] for x in sel) / len(sel), 4) if sel else None
    qa = [x for x in recs if x["set"] == "qa"]; km = [x for x in recs if x["set"] == "kmmlu"]
    s = {"model": model, "qa_acc": acc(qa), "kmmlu_acc": acc(km), "all_acc": acc(recs),
         "json_contamination": round(sum(x["json_like"] for x in recs) / len(recs), 4),
         "qa_by_category": {c: acc([x for x in qa if x["category"] == c]) for c in dict.fromkeys(x["category"] for x in qa)},
         "kmmlu_by_subject": {c: acc([x for x in km if x["category"] == c]) for c in dict.fromkeys(x["category"] for x in km)},
         "n_qa": len(qa), "n_kmmlu": len(km), "minutes": round((time.time() - t0) / 60, 1),
         "run_at": time.strftime("%Y-%m-%d %H:%M")}
    summary.append(s)
    tag = model.replace(":", "_").replace("/", "_")
    json.dump({"summary": s, "records": recs}, open(OUT / f"regression_{tag}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(s, ensure_ascii=False))

# ---------- 비교표 ----------
lines = ["| 모델 | 일반 질문 30 | KMMLU 50 | 전체 80 | 추출 JSON으로 답한 비율 |", "|---|---|---|---|---|"]
for s in summary:
    lines.append(f"| {s['model']} | {s['qa_acc']:.1%} | {s['kmmlu_acc']:.1%} | {s['all_acc']:.1%} | {s['json_contamination']:.1%} |")
if len(summary) >= 2:
    b, f = summary[0], summary[-1]
    d = (f["all_acc"] - b["all_acc"]) * 100
    lines.append("")
    lines.append(f"전체 정답률 변화: {d:+.1f}%p → " + ("기존 능력 유지 (하락 3%p 이내)" if d >= -3 else "기존 능력 하락 (3%p 초과)"))
    cats = list(b["qa_by_category"])
    lines += ["", "| 분류 | " + " | ".join(x["model"] for x in summary) + " |", "|---|" + "---|" * len(summary)]
    for c in cats:
        lines.append(f"| {c} | " + " | ".join(f"{x['qa_by_category'].get(c, 0):.0%}" for x in summary) + " |")
    for c in b["kmmlu_by_subject"]:
        lines.append(f"| KMMLU {c} | " + " | ".join(f"{x['kmmlu_by_subject'].get(c, 0):.0%}" for x in summary) + " |")
md = "\n".join(lines)
(OUT / "regression_summary.md").write_text(md + "\n", encoding="utf-8")
print("\n" + md)
