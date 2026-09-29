#!/usr/bin/env bash
# 등록한 모델 전체를 GPU(300건)와 CPU(50건)로 평가함. 서비스와 같은 조건(스키마 강제, temperature 0).
# 실행 (레포 맨 위): bash ml/eval/run_all.sh          끝나면: python ml/eval/compare.py
# 절전 모드를 꺼 두고 실행. 중간에 끊겨도 다시 실행하면 이미 끝난 측정은 건너뜀
set -u
cd "$(dirname "$0")/../.."

MODELS=(                       # ollama list에 보이는 이름 그대로. 빼거나 더할 때 여기만 고침
  itda-gemma4-e4b:q4_k_m
  itda-gemma4-e4b:q8_0
  itda-gemma3-4b:q4_k_m
  itda-qwen:q4_k_m
  itda-exaone4-1.2b:q8_0
)
CPU_N=50

run() {  # $1 모델, $2 gpu|cpu
  local tag="${1//:/_}_$2"                                  # 파일 이름에 콜론을 쓰지 않음 (Windows에서 clone 불가)
  if [ -f "ml/eval/results/eval_${tag}.json" ]; then echo "건너뜀 (이미 있음): $tag"; return; fi
  echo "===== $(date +%H:%M) $tag"
  for m in $(ollama ps | awk 'NR>1{print $1}'); do ollama stop "$m"; done   # 올라가 있는 모델을 모두 내림
  if [ "$2" = gpu ]; then
    python ml/eval/eval_ollama.py --model "$1" --tag "$tag" --show 0 | tail -22
  else
    python ml/eval/eval_ollama.py --model "$1" --tag "$tag" --show 0 --cpu --n "$CPU_N" | tail -22
  fi
}

for m in "${MODELS[@]}"; do run "$m" gpu; done     # GPU 먼저 전부
for m in "${MODELS[@]}"; do run "$m" cpu; done     # 그다음 CPU 전부
echo "===== $(date +%H:%M) 끝. python ml/eval/compare.py 로 비교표 확인"
