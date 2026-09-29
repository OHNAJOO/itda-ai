"""평가 질문 1 (학습은 안정적인가): 학습 기록을 W&B에 올리고 발표용 그래프 PNG를 만듦. 재학습은 필요 없음.
학습 때 report_to="none"이었지만, 체크포인트의 trainer_state.json에 같은 기록(loss, eval_loss, grad_norm,
learning_rate)이 남아 있어 그대로 W&B에 옮김.

준비 (한 번): uv pip install wandb matplotlib && wandb login     (API 키는 https://wandb.ai/authorize)
실행 (레포 맨 위):
  python ml/train/log_to_wandb.py --out ml/train/out_gemma_e4B --name gemma4-e4b-qlora \
      --model unsloth/gemma-4-E4B-it --lora ml/train/out_gemma_e4B/lora_itda \
      --eval ml/eval/results/eval_before_test_gemma4_E4B.json ml/eval/results/eval_itda-gemma4-e4b_q4_k_m_gpu.json
옵션:
  --state    trainer_state.json을 직접 지정 (없으면 --out 아래 체크포인트 중 가장 마지막 것을 씀)
  --no-wandb W&B에 올리지 않고 그래프 PNG만 만듦
결과: W&B 프로젝트 itda-model-a의 run 1개 + ml/eval/results/train_curves_<name>.png
"""
import argparse, glob, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", help="학습 출력 폴더 (예: ml/train/out_gemma_e4B)")
ap.add_argument("--state", help="trainer_state.json 경로")
ap.add_argument("--name", required=True, help="W&B run 이름 (예: gemma4-e4b-qlora)")
ap.add_argument("--project", default="itda-model-a")
ap.add_argument("--entity", default=None, help="팀 W&B 계정이 있으면 지정")
ap.add_argument("--model", default=None, help="베이스 모델 이름 (config 기록용)")
ap.add_argument("--lora", default=None, help="LoRA 폴더 (adapter_config.json에서 r, alpha, 대상 층을 읽음)")
ap.add_argument("--eval", nargs="*", default=[], help="같이 올릴 평가 결과 json (Golden Set 결과표)")
ap.add_argument("--no-wandb", action="store_true")
args = ap.parse_args()

# ---- trainer_state.json 찾기: 마지막 체크포인트에 전체 기록이 들어 있음 ----
if args.state:
    state_path = Path(args.state)
else:
    cands = [Path(p) for p in glob.glob(str(Path(args.out) / "**" / "trainer_state.json"), recursive=True)]
    if not cands:
        raise SystemExit(f"trainer_state.json을 찾지 못함: {args.out}")
    state_path = max(cands, key=lambda p: json.load(open(p)).get("global_step", 0))
state = json.load(open(state_path, encoding="utf-8"))
print("학습 기록:", state_path, "| 스텝", state.get("global_step"), "| 기록", len(state["log_history"]), "줄")

# ---- 스텝별로 모으기 ----
KEYMAP = {"loss": "train/loss", "grad_norm": "train/grad_norm", "learning_rate": "train/learning_rate",
          "epoch": "train/epoch", "eval_loss": "eval/loss", "eval_runtime": "eval/runtime",
          "eval_samples_per_second": "eval/samples_per_second", "eval_steps_per_second": "eval/steps_per_second"}
by_step, final = {}, {}
for rec in state["log_history"]:
    step = rec.get("step")
    if "train_runtime" in rec:          # 학습 끝에 한 번 남는 요약 줄
        final = rec; continue
    row = by_step.setdefault(step, {})
    for k, v in rec.items():
        if k in KEYMAP and v is not None:
            row[KEYMAP[k]] = v
steps = sorted(by_step)
train = [(s, by_step[s]["train/loss"]) for s in steps if "train/loss" in by_step[s]]
evals = [(s, by_step[s]["eval/loss"]) for s in steps if "eval/loss" in by_step[s]]
gnorm = [(s, by_step[s]["train/grad_norm"]) for s in steps if "train/grad_norm" in by_step[s]]
lrs = [(s, by_step[s]["train/learning_rate"]) for s in steps if "train/learning_rate" in by_step[s]]
best_step = None
if state.get("best_model_checkpoint"):
    try: best_step = int(str(state["best_model_checkpoint"]).rstrip("/").split("-")[-1])
    except ValueError: pass

# ---- 과적합·불안정 자동 점검 (W&B 화면을 읽는 기준을 숫자로) ----
checks = {}
if evals:
    e = [v for _, v in evals]; low = min(range(len(e)), key=e.__getitem__)
    checks["eval_loss_min"] = round(e[low], 4); checks["eval_loss_min_step"] = evals[low][0]
    checks["eval_loss_last"] = round(e[-1], 4)
    checks["eval_rise_after_min_pct"] = round((e[-1] - e[low]) / e[low] * 100, 1) if e[low] else None
if train:
    checks["train_loss_first"] = round(train[0][1], 4); checks["train_loss_last"] = round(train[-1][1], 4)
if gnorm:
    g = [v for _, v in gnorm]
    checks["grad_norm_max"] = round(max(g), 3); checks["grad_norm_median"] = round(sorted(g)[len(g) // 2], 3)
    checks["grad_norm_nan"] = sum(1 for v in g if v != v)
print("점검:", json.dumps(checks, ensure_ascii=False))

# ---- config ----
cfg = {"base_model": args.model, "trainer_state": str(state_path), "global_step": state.get("global_step"),
       "num_train_epochs": state.get("num_train_epochs"), "train_batch_size": state.get("train_batch_size"),
       "eval_steps": state.get("eval_steps"), "logging_steps": state.get("logging_steps"),
       "best_model_checkpoint": state.get("best_model_checkpoint"), "best_eval_loss": state.get("best_metric")}
if args.lora and (Path(args.lora) / "adapter_config.json").exists():
    ac = json.load(open(Path(args.lora) / "adapter_config.json"))
    cfg.update({"lora_r": ac.get("r"), "lora_alpha": ac.get("lora_alpha"), "lora_dropout": ac.get("lora_dropout"),
                "target_modules": str(ac.get("target_modules"))[:300]})

# ---- 평가 결과 (Golden Set 결과표) ----
eval_rows = []
for p in args.eval:
    s = json.load(open(p, encoding="utf-8"))["summary"]
    eval_rows.append([Path(p).stem.removeprefix("eval_"), s.get("model", "(노트북)"), s.get("schema", "none"),
                      s.get("n"), s.get("f1"), s.get("precision"), s.get("recall"), s.get("exact_match_rate"),
                      s.get("json_ok_rate"), s.get("empty_correct_rate"), s.get("sec_per_memo")])

# ---- 그래프 PNG (발표 자료용, W&B 화면과 같은 내용) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
kr = next((f.name for f in font_manager.fontManager.ttflist
           if any(k in f.name for k in ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic", "AppleGothic"])), None)
if kr: plt.rcParams["font.family"] = kr
plt.rcParams["axes.unicode_minus"] = False
T = (lambda ko, en: ko) if kr else (lambda ko, en: en)
BLUE, ORANGE = "#2a78d6", "#eb6834"                 # 검증한 2색 (색각 이상에서도 구분됨)
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor="#fcfcfb")
def style(ax, title):
    ax.set_facecolor("#fcfcfb"); ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]: ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9); ax.set_xlabel("step", color=MUTED, fontsize=9)
ax = axes[0]; style(ax, T("손실: 학습 vs 검증", "Loss: train vs eval"))
if train: ax.plot(*zip(*train), color=BLUE, linewidth=2, label="train/loss")
if evals: ax.plot(*zip(*evals), color=ORANGE, linewidth=2, marker="o", markersize=6,
                  markeredgecolor="#fcfcfb", markeredgewidth=1.5, label="eval/loss")
if best_step is not None:
    ax.axvline(best_step, color=AXIS, linestyle="--", linewidth=1)
    ax.annotate(T(f"최저 검증 손실 (step {best_step})", f"best eval (step {best_step})"), (best_step, ax.get_ylim()[1]),
                xytext=(4, -14), textcoords="offset points", color=MUTED, fontsize=8)
ax.legend(frameon=False, fontsize=9, labelcolor=INK)
if gnorm:
    ax = axes[1]; style(ax, "grad_norm"); ax.plot(*zip(*gnorm), color=BLUE, linewidth=2)
if lrs:
    ax = axes[2]; style(ax, "learning_rate"); ax.plot(*zip(*lrs), color=BLUE, linewidth=2)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
fig.suptitle(f"{args.name}", x=0.01, ha="left", color=INK, fontsize=13)
fig.tight_layout()
png = Path("ml/eval/results") / f"train_curves_{args.name}.png"
png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(png, dpi=160, facecolor=fig.get_facecolor()); print("그래프:", png)

# ---- W&B ----
if args.no_wandb:
    raise SystemExit(0)
import wandb
run = wandb.init(project=args.project, entity=args.entity, name=args.name, config=cfg, job_type="train",
                 tags=["qlora", "replay-from-trainer_state"],
                 notes="학습 때 기록한 trainer_state.json을 그대로 옮긴 run (값은 학습 당시 기록과 같음)")
wandb.define_metric("train/global_step")
wandb.define_metric("train/*", step_metric="train/global_step")
wandb.define_metric("eval/*", step_metric="train/global_step")
for s in steps:
    row = dict(by_step[s]); row["train/global_step"] = s
    wandb.log(row)
for k, v in final.items():
    if k not in ("step", "epoch"): run.summary[f"train/{k}"] = v
for k, v in checks.items():
    run.summary[f"check/{k}"] = v
if eval_rows:
    cols = ["결과", "모델", "스키마", "건수", "F1", "정밀도", "재현율", "메모완전일치", "JSON통과", "빈결과정확", "초/건"]
    run.log({"golden_set/results": wandb.Table(columns=cols, data=eval_rows)})
    for r in eval_rows:
        run.summary[f"golden/{r[0]}/f1"] = r[4]
run.log({"train_curves": wandb.Image(str(png))})
url = run.get_url() if hasattr(run, "get_url") else run.url
run.finish()
print("W&B:", url)
