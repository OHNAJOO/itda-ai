"""LoRA를 16비트 원본 모델에 직접 합쳐 새 폴더에 저장함 (Unsloth 합치기 우회).
원본을 조각(shard)별로 읽어 합치므로 RAM을 적게 쓰고, Hugging Face 캐시 원본은 건드리지 않음.

실행 (itda-ai 레포 맨 위에서, 노트북 커널은 꺼 두기):
  python ml/train/merge_lora.py --base unsloth/gemma-4-E4B-it --lora ml/train/out_gemma_e4B/lora_itda --out ml/train/out_gemma_e4B/merged
"""
import argparse, json, re, shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="Hugging Face 모델 이름 또는 로컬 폴더 (16비트 원본)")
ap.add_argument("--lora", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

base = Path(args.base)
if not base.exists():
    from huggingface_hub import snapshot_download
    base = Path(snapshot_download(args.base, allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.model", "tokenizer*"]))
lora_dir, out = Path(args.lora), Path(args.out)
out.mkdir(parents=True, exist_ok=True)

cfg = json.load(open(lora_dir / "adapter_config.json"))
r = cfg["r"]; alpha = cfg.get("lora_alpha", r)
scale = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r
print(f"LoRA r={r}, alpha={alpha}, scale={scale}")

# LoRA 가중치를 모듈 이름 끝부분(layers.N....proj) 기준으로 모음
lora = {}
with safe_open(str(lora_dir / "adapter_model.safetensors"), "pt") as f:
    for k in f.keys():
        if "vision" in k or "audio" in k:   # 언어 모델 층만 사용
            continue
        m = re.search(r"(layers\.\d+\..+?)\.lora_([AB])(?:\.default)?\.weight$", k)
        if not m:
            continue
        lora.setdefault(m.group(1), {})[m.group(2)] = f.get_tensor(k).float()
print("LoRA 모듈 수:", len(lora))

def lora_key_for(base_key):
    """원본 키 중 언어 모델 층의 weight만 LoRA와 연결 (비전·오디오 층은 제외)"""
    if "vision" in base_key or "audio" in base_key or not base_key.endswith(".weight"):
        return None
    m = re.search(r"(layers\.\d+\..+)\.weight$", base_key)
    return m.group(1) if m and m.group(1) in lora else None

shards = sorted(base.glob("*.safetensors"))
merged_count, index = 0, {"metadata": {}, "weight_map": {}}
for sh in shards:
    tensors = {}
    with safe_open(str(sh), "pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            lk = lora_key_for(k)
            if lk is not None:
                A, B = lora[lk]["A"], lora[lk]["B"]
                t = (t.float() + scale * (B @ A)).to(t.dtype)
                merged_count += 1
            tensors[k] = t.contiguous()
    save_file(tensors, str(out / sh.name), metadata={"format": "pt"})
    for k in tensors: index["weight_map"][k] = sh.name
    print(f"  {sh.name}: 저장 ({len(tensors)}개 텐서)")
    del tensors

if len(shards) > 1:
    json.dump(index, open(out / "model.safetensors.index.json", "w"), indent=1)
for p in base.iterdir():   # 설정·토크나이저 파일 복사
    if p.is_file() and p.suffix != ".safetensors" and p.name != "model.safetensors.index.json":
        shutil.copy(p, out / p.name)
print(f"합친 층: {merged_count} / LoRA 모듈 {len(lora)}")
if merged_count != len(lora):
    print("경고: 합쳐지지 않은 LoRA 모듈이 있음. 이름 대응을 확인해야 함")
print("완료:", out)
