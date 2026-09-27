# itda-ai

치매 환자 보호자의 관찰 메모에서 사건(증상)을 뽑아내는 모델을 위한 데이터 생성/검증 파이프라인.

## 환경 설정 (uv)

의존성 관리와 실행에 [uv](https://docs.astral.sh/uv/)를 사용한다.

```bash
# uv 설치 (이미 설치돼 있으면 생략)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치 (.venv 생성 + uv.lock 기준 동기화)
uv sync

# 스크립트 실행은 항상 uv run으로
uv run python ml/data_gen/make_plans.py
```

새 패키지가 필요하면 `uv add <패키지명>`으로 추가한다 (`pyproject.toml`, `uv.lock`이 함께 갱신됨).

Python 3.12 이상, 의존성은 `anthropic`(Claude API 호출용), `pyyaml`(personas.yaml 로딩용) 두 개.

## 디렉터리

```
config/
  labels.json         # 증상 유형(type)·상태 등 코드→한글 라벨 매핑
  system_prompt.txt    # verbalize.py가 호출하는 Claude 문장화용 시스템 프롬프트
ml/data_gen/
  common.py           # 공통 상수(TYPES, KO 등)와 유틸 함수
  make_plans.py        # 계획 파일(S1~S4, W_val, W_test) 생성 — 메모마다 담을 증상 정답 목록
  personas.yaml         # 문체(T01~T10, V01, E01, E02) 정의
  make_batches.py       # 계획+문체를 묶어 배치 프롬프트 생성
  verbalize.py           # Claude API로 계획을 실제 메모 문장으로 변환
  add_answer.py          # 채팅 방식으로 받은 Claude 답을 저장, --drop으로 잘못된 건 제거
  check.py               # 결과 검증 및 눈 검사(육안 검토) 판정 관리
  merge.py               # 슬롯별 결과를 최종 데이터셋으로 병합
  plans/                # 슬롯별 계획 파일(S1~S4.json, W_val.json, W_test.json)
```

## 작업 흐름

1. `make_plans.py` — 슬롯별 계획 파일 생성 (이미 `plans/`에 있으면 보통 재실행 불필요)
2. `make_batches.py --slot <슬롯> --personas <문체목록> [--size N]` — 배치 프롬프트 생성
3. `verbalize.py --slot <슬롯> --personas <문체목록> [--limit N] [--redo]` — Claude로 문장화
   - 채팅으로 직접 답을 받는 경우엔 `add_answer.py --slot <슬롯> [--file 경로] [--drop 번호,번호]` 사용
4. `check.py --slot <슬롯> --personas <문체목록> --target <목표건수> [--review N]` — 검증 및 눈 검사
5. `merge.py` — 전체 슬롯 병합

## 담당 배정 (v1.2 기준)

| 담당 | 슬롯 | 문체 |
|---|---|---|
| 변은아 | S1 | T01, T07 |
| 최태순 | S2 | T02, T04 |
| 이주영 | S3 | T03, T10 |
| 조성률 | S4 | T05, T06 |
| 김우진 | W_val | V01 |
| 김우진 | W_test | E01, E02 |

자세한 개발 가이드는 `먼저읽기.txt`와 "잇다_상세개발가이드" 문서를 참고.
