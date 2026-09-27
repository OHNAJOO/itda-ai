"""데이터 생성 스크립트들이 함께 쓰는 설정과 함수. 이 파일은 고치지 않음."""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # itda-ai 레포 맨 위 폴더 (config/가 있는 곳)
HERE = Path(__file__).resolve().parent               # ml/data_gen/
LABELS = json.load(open(ROOT / "config" / "labels.json", encoding="utf-8"))
TYPES = list(LABELS["type"])                         # 영문 코드 12개, 기획안 4-3 표 순서
KO = LABELS["type"]                                  # 코드 → 한글 이름

LLM_MODEL = "claude-sonnet-5"                        # 문장화에 쓰는 모델. 전원 같은 값을 씀
PROMPT_VERSION = "v1"                                # 프롬프트를 바꾸면 김우진이 올리고 전원에게 알림

# 식사량 감소는 줄었다는 표현이 evidence에 있어야 함 (비교형 규칙, 기획안 4-3)
REDUCED_WORDS = ["평소보다", "평소의", "밖에", "만 드", "만 먹", "만 하", "숟갈", "숟가락", "남기", "남김",
                 "적게", "덜 ", "덜드", "덜 드", "안 드", "안드", "안 먹", "안먹", "거부", "거의", "쪼매", "조금만"]

def load_personas():
    import yaml
    return yaml.safe_load(open(HERE / "personas.yaml", encoding="utf-8"))

def assign_persona(index, personas):
    """계획 번호로 문체를 정함. 문체가 2개면 짝수 번은 첫째, 홀수 번은 둘째"""
    return personas[index % len(personas)]

def event_line(e):
    st = "있었음" if e["status"] == "present" else "없었음"
    te = e["time_expr"] if e["time_expr"] else "없음"
    cnt = f"{e['count']}회" if e["status"] == "present" else "-"
    return f"- {KO[e['type']]} / {st} / 시간 표현: {te} / 횟수: {cnt}"

def persona_block(p):
    return "\n".join(f"{k}: {v}" for k, v in p.items())

RULES = """규칙:
- 사건 목록에 있는 것만 증상으로 쓴다. 목록 밖의 증상은 쓰지 않는다
- 시간 표현이 주어지면 그 표현을 메모에 글자 그대로 넣는다. "없음"이면 시간 표현을 쓰지 않는다
- "없었음"은 "오늘은 ~는 없었음", "~하지 않으셨음"처럼 분명하게 부정한다
- 횟수가 2회 이상이면 "두 번", "세 번"처럼 횟수를 쓴다
- 식사량 감소는 "평소보다", "~밖에", "몇 숟갈만", "반도 안"처럼 줄었다는 표현을 반드시 넣는다
- 복약 거부는 환자가 약을 거부하거나 미룬 모습으로 쓴다 (깜빡 잊은 것이 아님)
- 사람·장소 혼동은 사람을 잘못 알아보거나 시간·장소를 헷갈린 모습으로 쓴다
- "함께 섞을 문장"은 한 문장으로 쓰되, 위 사건 목록의 증상처럼 보이지 않게 쓴다
- 사건 목록이 비어 있으면 증상과 무관한 일상 이야기만 쓴다
- 의학 용어(섬망, BPSD, 치매 단계 등)와 이모지는 쓰지 않는다
- 문체 설정의 말투, 길이, 습관, 오타를 따른다"""

def build_prompt(plan, persona):
    events = "\n".join(event_line(e) for e in plan["events"]) or "(없음. 일상 이야기만 쓴다)"
    distract = "\n".join(f"- {d}" for d in plan["distractors"]) or "(없음)"
    return f"""너는 치매 환자를 돌보는 가족 보호자다. 아래 문체로 관찰 메모 하나를 쓴다.

[문체]
{persona_block(persona)}

[메모에 반드시 담을 사건] 한글 유형 / 있었음·없었음 / 시간 표현 / 횟수
{events}

[함께 섞을 문장]
{distract}

{RULES}

[답하는 형식] 아래 JSON 한 줄로만 답한다. 설명은 쓰지 않는다.
{{"memo": "메모 전체", "evidence": ["사건1의 근거 구절", "사건2의 근거 구절"]}}
- evidence는 사건 목록과 같은 순서, 같은 개수로 쓴다. 사건이 없으면 []
- 각 근거 구절은 memo 안에 글자 하나 다르지 않게 그대로 들어 있어야 한다
- 근거 구절은 그 사건을 보여 주는 짧은 부분(보통 5~25자)이다"""

def parse_json_obj(text):
    """모델 답에서 첫 '{'부터 마지막 '}'까지 잘라 JSON으로 읽음"""
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < 0:
        raise ValueError("JSON 없음")
    return json.loads(text[a:b + 1])

def read_jsonl(path):
    rows = []
    if Path(path).exists():
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
