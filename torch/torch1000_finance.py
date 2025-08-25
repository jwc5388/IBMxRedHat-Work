# run_baseline_qwen2.py
# 단일 LLM(Qwen2-7B-Instruct) 오프라인 추론 베이스라인
import os, re, argparse, pandas as pd, torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def detect_mcq_options(question: str):
    """
    줄 시작의 숫자-선지 패턴 탐지. 예) \n1 ... \n2 ... 형태
    반환: ['1','2',...], [(번호, 텍스트), ...]
    """
    opts = re.findall(r"(?:^|\n)\s*([0-9])\s([^\n]+)", question)
    digits = sorted({d for d,_ in opts})
    return digits, opts

def build_prompt_mcq(q: str, digits):
    last = digits[-1]
    return (
        "다음은 객관식 문제다. 질문과 선택지를 읽고 정답 번호만 출력하라.\n"
        f"정답번호는 {digits[0]}~{last} 중 하나이며, **숫자 한 글자만** 출력한다.\n\n"
        f"{q}\n\n정답번호:"
    )

def build_prompt_subj(q: str):
    return (
        "아래 질문에 대해 한 줄 핵심답과 키워드를 출력하라.\n"
        "형식:\n한줄정답: <핵심문장>\n키워드: <키1>, <키2>, <키3>\n\n"
        f"질문:\n{q}\n"
    )

def load_model(model_path: str):
    # 4bit 양자화(NF4) 우선 시도 → 실패 시 fp16/bf16 로딩
    try:
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     quantization_config=bnb,
                                                     device_map="auto",
                                                     torch_dtype=torch.bfloat16)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     device_map="auto",
                                                     torch_dtype=torch.bfloat16)
    return tok, model

@torch.no_grad()
def choose_digit_by_logits(tok, model, prompt, digit_list):
    """다음 토큰 로짓에서 후보 숫자만 확률 비교."""
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model(**inputs)
    logits = out.logits[0, -1, :]
    cand = {d: tok.convert_tokens_to_ids(d) for d in digit_list}
    probs = torch.softmax(logits, dim=-1)
    best = max(digit_list, key=lambda d: float(probs[cand[d]]))
    return best

@torch.no_grad()
def sample_digit_by_generate(tok, model, prompt, digit_list, temperature=0.7):
    """샘플링으로 한 번 생성하고 가장 먼저 등장한 후보 숫자를 파싱."""
    ids = tok(prompt, return_tensors="pt").to(model.device)
    gen = model.generate(**ids, max_new_tokens=4, do_sample=True, temperature=temperature,
                         top_p=0.9, pad_token_id=tok.eos_token_id)
    out = tok.decode(gen[0], skip_special_tokens=True)
    tail = out[len(prompt):]
    m = re.search(r"([0-9])", tail)
    return m.group(1) if (m and m.group(1) in digit_list) else None

def postproc_mcq(digit, opts):
    # 정답: <번호> (<선지 텍스트 일부>)
    choice = next((t for d,t in opts if d==digit), "")
    return f"정답: {digit} ({choice.strip()[:60]})"

def postproc_subj(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "한줄정답: 핵심 요약\n키워드: 핵심, 키워드"
    if not lines[0].startswith("한줄정답:"):
        lines[0] = "한줄정답: " + lines[0]
    if len(lines) == 1:
        # 첫 줄에서 키워드 후보 추출
        words = re.findall(r"[가-힣A-Za-z0-9\-]+", lines[0])[:8]
        lines.append("키워드: " + ", ".join(words))
    if not lines[1].startswith("키워드:"):
        lines[1] = "키워드: " + lines[1]
    return "\n".join(lines[:2])

@torch.no_grad()
def gen_text(tok, model, prompt, max_new_tokens=96, temperature=0.25):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    gen = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=True,
                         temperature=temperature, top_p=0.9,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(gen[0], skip_special_tokens=True)[len(prompt):].strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--input_csv", default="data/test.csv")
    ap.add_argument("--output_csv", default="outputs/pred.csv")
    ap.add_argument("--mcq_samples", type=int, default=3)  # 1=로짓 단일결정, >1=다수결
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    tok, model = load_model(args.model_path)
    df = pd.read_csv(args.input_csv)
    outputs = []

    for _, r in df.iterrows():
        qid, q = r["ID"], r["Question"]
        digit_list, opts = detect_mcq_options(q)

        if digit_list:  # 객관식
            prompt = build_prompt_mcq(q, digit_list)
            if args.mcq_samples <= 1:
                digit = choose_digit_by_logits(tok, model, prompt, digit_list)
            else:
                votes = []
                # 로짓 한 번 + 샘플링 반복 → 안정성과 속도 균형
                votes.append(choose_digit_by_logits(tok, model, prompt, digit_list))
                for _ in range(args.mcq_samples - 1):
                    d = sample_digit_by_generate(tok, model, prompt, digit_list, temperature=0.7)
                    if d: votes.append(d)
                digit = Counter(votes).most_common(1)[0][0]
            final = postproc_mcq(digit, opts)

        else:  # 주관식
            prompt = build_prompt_subj(q)
            txt = gen_text(tok, model, prompt, max_new_tokens=120, temperature=0.25)
            final = postproc_subj(txt)

        outputs.append({"ID": qid, "Answer": final})

    pd.DataFrame(outputs).to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"saved -> {args.output_csv}")

if __name__ == "__main__":
    main()