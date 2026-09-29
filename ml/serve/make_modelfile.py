"""gguf 파일 안에 저장된 채팅 템플릿을 읽어, 학습 때와 같은 프롬프트가 되도록 Ollama Modelfile을 만듦.
모델마다 대화 형식이 달라서(Gemma, EXAONE, Qwen ...) 손으로 쓰지 않고 gguf에 든 원본 템플릿을 그대로 렌더링함.

준비 (한 번): uv pip install ~/llama.cpp/gguf-py jinja2
실행 (레포 맨 위):
  python ml/serve/make_modelfile.py --gguf ml/train/out_qwen/itda-qwen-q4_k_m.gguf --out ml/serve/Modelfile.qwen
  ollama create itda-qwen:q4_k_m -f ml/serve/Modelfile.qwen
옵션:
  --tokenizer  gguf에 템플릿이 없을 때 Hugging Face 토크나이저 이름으로 대신 가져옴 (예: LGAI-EXAONE/EXAONE-4.0-1.2B)
  --thinking   생각 모드를 켠 템플릿으로 만듦 (기본은 끔. 우리 학습은 모두 생각 모드를 끄고 JSON만 답하게 함)
"""
import argparse, json, datetime
from pathlib import Path
import jinja2, jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tokenizer", default=None)
ap.add_argument("--thinking", action="store_true")
ap.add_argument("--num-ctx", type=int, default=2048)
args = ap.parse_args()
gguf_path = Path(args.gguf).resolve()
assert gguf_path.exists(), f"gguf 없음: {gguf_path}"

# ---- gguf 메타데이터 읽기 ----
from gguf import GGUFReader
r = GGUFReader(str(gguf_path))
def val(key):
    f = r.fields.get(key)
    if f is None: return None
    if hasattr(f, "contents"):
        return f.contents()
    v = f.parts[f.data[0]]
    return bytes(v).decode("utf-8") if f.types and f.types[0].name == "STRING" else v.tolist()[0]
def token_text(key):
    tid = val(key)
    if tid is None: return None
    f = r.fields["tokenizer.ggml.tokens"]
    if hasattr(f, "contents"):
        return f.contents()[tid]
    return bytes(f.parts[f.data[tid]]).decode("utf-8")

print("모델:", val("general.name"), "|", val("general.architecture"), "|", val("general.size_label"))
template_src = val("tokenizer.chat_template")
bos, eos = token_text("tokenizer.ggml.bos_token_id"), token_text("tokenizer.ggml.eos_token_id")
if args.tokenizer:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    template_src, bos, eos = tok.chat_template, tok.bos_token, tok.eos_token
if not template_src:
    raise SystemExit("gguf에 채팅 템플릿이 없음. --tokenizer <Hugging Face 모델 이름>으로 다시 실행")
print("bos:", repr(bos), "| eos:", repr(eos))

# ---- transformers와 같은 방식으로 템플릿 렌더링 ----
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, separators=None, sort_keys=False: \
    json.dumps(x, indent=indent, ensure_ascii=ensure_ascii, separators=separators, sort_keys=sort_keys)
def raise_exception(msg): raise jinja2.exceptions.TemplateError(msg)
env.globals.update(raise_exception=raise_exception,
                   strftime_now=lambda fmt: datetime.datetime.now().strftime(fmt))
msgs = [{"role": "system", "content": "@@SYS@@"}, {"role": "user", "content": "@@USER@@"}]
rendered = env.from_string(template_src).render(
    messages=msgs, add_generation_prompt=True, enable_thinking=args.thinking,
    bos_token=bos or "", eos_token=eos or "", tools=None)
print("학습 때 프롬프트 모양:", repr(rendered))
assert rendered.count("@@SYS@@") == 1, "시스템 프롬프트 자리가 하나가 아님 (템플릿이 system을 버렸는지 확인 필요)"
assert rendered.count("@@USER@@") == 1
if bos and rendered.startswith(bos):
    rendered = rendered[len(bos):]          # BOS는 Ollama(llama.cpp)가 자동으로 붙이므로 빼야 두 번 들어가지 않음
assert "{{" not in rendered and '"""' not in rendered
template = rendered.replace("@@SYS@@", "{{ .System }}").replace("@@USER@@", "{{ .Prompt }}")

# 멈춤 토큰: eos + 템플릿에 나오는 턴 끝 표시
stops = [s for s in [eos, "<end_of_turn>", "<|im_end|>", "[|endofturn|]", "<turn|>", "<|eot_id|>"]
         if s and (s == eos or s in template_src)]
stops = list(dict.fromkeys(stops))

out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
lines = [f"FROM {gguf_path}", f'TEMPLATE """{template}"""']
lines += [f'PARAMETER stop "{s}"' for s in stops]
lines += ["PARAMETER temperature 0", f"PARAMETER num_ctx {args.num_ctx}"]
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("저장:", out, "\n" + out.read_text(encoding="utf-8"))
