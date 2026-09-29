"""Ollama에 등록한 모델로 평가 ①을 돌림. 채점 방식은 학습 노트북의 evaluate()와 같음 (숫자를 바로 비교할 수 있게).

실행 (레포 맨 위, Ollama가 켜져 있어야 함):
  python ml/eval/eval_ollama.py --model itda-gemma4-e4b:q4_k_m --tag itda-gemma4-e4b_q4_k_m_gpu
옵션:
  --file      평가 파일 (기본 ml/data/synth_eval.jsonl, val.jsonl이나 수작업 파일도 가능)
  --n         앞에서 n건만
  --no-schema JSON 스키마 강제 없이 호출 (노트북 평가와 같은 조건)
  --cpu       GPU를 쓰지 않고 잼 (평가 ⑤ 추론 시간, options.num_gpu = 0)
결과: ml/eval/results/eval_<tag>.json  (summary + 건별 기록). 끝나면 모델을 메모리에서 내림

지표
  precision/recall/f1   사건 단위. 같은 메모 안에서 (유형, 있었음/없었음)이 같으면 맞은 사건 (기획안 8-4 ①)
  exact_match_rate      메모 단위 완전 일치율. 메모 하나의 사건 목록(유형·있었음/없었음)이 정답과 통째로 같은 비율
  json_ok_rate          JSON으로 읽힌 비율
  empty_correct_rate    사건이 없는 메모에서 빈 결과를 낸 비율
  time_expr_match_rate  맞은 사건 중 시간 표현까지 같은 비율
  evidence_in_memo_rate 예측 evidence가 메모 원문에 그대로 들어 있는 비율
  gpu_fraction          Ollama가 모델을 GPU 메모리에 올린 비율 (1.0 = 전부 GPU, 0 = 전부 CPU)
  server_errors         Ollama가 오류를 돌려준 건수 (재시도 1회 뒤에도 실패). 해당 메모는 JSON 실패로 채점
"""
import argparse, json, os, subprocess, time, urllib.request
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--file", default="ml/data/synth_eval.jsonl")
ap.add_argument("--n", type=int, default=None)
ap.add_argument("--no-schema", action="store_true")
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--host", default="http://localhost:11434")
ap.add_argument("--show", type=int, default=3)
args = ap.parse_args()

ROOT = Path.cwd()
SYSTEM = (ROOT / "config" / "system_prompt.txt").read_text(encoding="utf-8").strip()
schema_path = ROOT / "config" / "event_schema.json"
if args.no_schema:
    FORMAT = None
elif schema_path.exists():
    FORMAT = json.loads(schema_path.read_text(encoding="utf-8"))
else:
    raise SystemExit("config/event_schema.json이 없음. 스키마를 넣거나 --no-schema로 실행")
OPTIONS = {"temperature": 0, "num_predict": 384, **({"num_gpu": 0} if args.cpu else {})}

class OllamaError(Exception): pass

def post(path, body, timeout=600):
    req = urllib.request.Request(args.host + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        msg = f"Ollama 오류 {e.code}: {e.read().decode('utf-8', 'ignore')}"
        if e.code == 404:
            raise SystemExit(msg + f"  (모델 이름 확인: {args.model})")
        raise OllamaError(msg)
    except (urllib.error.URLError, TimeoutError) as e:
        raise OllamaError(f"연결 오류: {e}")

def chat(msgs, retry=1):
    """메모 한 건 호출. 서버 오류가 나면 한 번 더 시도하고, 그래도 실패하면 오류 문자열을 답으로 돌려줌 (평가는 계속)"""
    body = {"model": args.model, "messages": msgs, "stream": False, "keep_alive": "30m", "options": OPTIONS}
    if FORMAT is not None:
        body["format"] = FORMAT
    for k in range(retry + 1):
        try:
            return post("/api/chat", body)["message"]["content"]
        except OllamaError as e:
            err = str(e)
    return "<<ERROR>> " + err

def gpu_fraction():
    try:
        with urllib.request.urlopen(args.host + "/api/ps", timeout=10) as resp:
            for m in json.loads(resp.read()).get("models", []):
                if m.get("name") in (args.model, args.model + ":latest") or m.get("model") == args.model:
                    return round(m.get("size_vram", 0) / m["size"], 3) if m.get("size") else None
    except Exception:
        return None

def env_info():
    info = {"cpu": None, "cpu_threads": os.cpu_count(), "gpu": None}
    try:
        info["cpu"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception: pass
    try:
        info["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception: pass
    return info

def parse_events(text):
    """모델 답에서 JSON을 찾아 events 목록을 돌려줌. 못 읽으면 None"""
    t = text.strip().replace("```json", "").replace("```", "")
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b < 0: return None
    try:
        obj = json.loads(t[a:b + 1])
        ev = obj.get("events") if isinstance(obj, dict) else None
        return ev if isinstance(ev, list) else None
    except Exception:
        return None

def load_rows(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.strip(): continue
        r = json.loads(line)
        if "messages" in r:      # train/val 형식
            memo = r["messages"][1]["content"]; gold = json.loads(r["messages"][2]["content"])["events"]
        else:                    # synth_eval, 수작업 형식
            memo = r["memo"]; gold = r["gold"]["events"]
        rows.append((memo, gold))
    return rows[:args.n]

rows = load_rows(ROOT / args.file)
chat([{"role": "user", "content": "안녕"}])   # 모델을 미리 올려 두어 첫 건 시간이 튀지 않게 함
gfrac = gpu_fraction()
print(f"모델 {args.model} | {'CPU 전용' if args.cpu else 'GPU'} 요청 | GPU에 올라간 비율 {gfrac} | {len(rows)}건", flush=True)

tp = fp = fn = 0; json_ok = 0; exact = 0; n_err = 0; empty_total = empty_ok = 0; te_match = te_total = 0
ev_total = ev_in = 0
per_type, records, times = {}, [], []
for i, (memo, gold) in enumerate(rows):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": memo}]
    t0 = time.time(); raw = chat(msgs); times.append(time.time() - t0)
    if raw.startswith("<<ERROR>>"):
        n_err += 1; print(f"  [{i}] 서버 오류로 실패 처리: {raw[10:160]}", flush=True)
    pred = parse_events(raw)
    ok = pred is not None
    if ok: json_ok += 1
    pred = [e for e in (pred or []) if isinstance(e, dict)]
    g = Counter((e["type"], e["status"]) for e in gold)
    p = Counter((e.get("type"), e.get("status")) for e in pred)
    hit = g & p
    tp += sum(hit.values()); fp += sum((p - g).values()); fn += sum((g - p).values())
    exact += int(ok and g == p)
    for (t, _), c in g.items(): per_type.setdefault(t, [0, 0, 0]); per_type[t][2] += c
    for (t, _), c in p.items(): per_type.setdefault(t, [0, 0, 0]); per_type[t][1] += c
    for (t, _), c in hit.items(): per_type[t][0] += c
    if not gold:
        empty_total += 1; empty_ok += int(ok and len(pred) == 0)
    pd = {(e.get("type"), e.get("status")): e.get("time_expr") for e in pred}
    for e in gold:
        k = (e["type"], e["status"])
        if k in pd: te_total += 1; te_match += int(pd[k] == e["time_expr"])
    for e in pred:
        ev = e.get("evidence")
        if isinstance(ev, str) and ev.strip():
            ev_total += 1; ev_in += int(ev.strip() in memo)
    records.append({"memo": memo, "gold": gold, "pred_raw": raw, "sec": round(times[-1], 2)})
    if i < args.show:
        print(f"[{i}] 메모: {memo}\n    예측: {raw.strip()[:300]}\n    정답: {json.dumps(gold, ensure_ascii=False)[:300]}\n")
    if (i + 1) % 25 == 0:
        print(f"  {i + 1}/{len(rows)}건, {round(sum(times))}초", flush=True)

post("/api/generate", {"model": args.model, "keep_alive": 0})   # 다음 측정에 영향이 없도록 모델을 내림

prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0
f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
st = sorted(times)
res = {"tag": args.tag, "model": args.model, "file": args.file, "n": len(rows),
       "schema": "none" if args.no_schema else "event_schema",
       "device": "cpu" if args.cpu else "gpu", "gpu_fraction": gfrac,
       "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
       "exact_match_rate": round(exact / len(rows), 4),
       "json_ok_rate": round(json_ok / len(rows), 4), "server_errors": n_err,
       "empty_correct_rate": round(empty_ok / empty_total, 4) if empty_total else None,
       "time_expr_match_rate": round(te_match / te_total, 4) if te_total else None,
       "evidence_in_memo_rate": round(ev_in / ev_total, 4) if ev_total else None,
       "per_type_f1": {t: round(2 * v[0] / (v[1] + v[2]), 3) if v[1] + v[2] else None for t, v in sorted(per_type.items())},
       "sec_per_memo": round(sum(times) / len(times), 2), "sec_median": round(st[len(st) // 2], 2),
       "sec_max": round(max(times), 2), "env": env_info(),
       "run_at": time.strftime("%Y-%m-%d %H:%M")}
out = ROOT / "ml" / "eval" / "results"; out.mkdir(parents=True, exist_ok=True)
json.dump({"summary": res, "records": records}, open(out / f"eval_{args.tag}.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "per_type_f1"}, ensure_ascii=False, indent=1))
