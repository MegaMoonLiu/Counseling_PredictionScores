import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM, PeftModel

EVAL_ITEM_TITLES: List[str] = [
    "聴いてもらえた、わかってもらえたと感じた",
    "尊重されたと感じた",
    "新しい気づきや体験があった",
    "希望や期待を感じられた",
    "取り組みたかったことを扱えた",
    "一緒に考えながら取り組めた",
    "やりとりのリズムがあっていた",
    "居心地のよいやりとりだった",
    "全体として適切でよかった",
    "今回の相談は価値があった",
    "相談開始の円滑さ",
    "相談終了のタイミング（不必要に聴きすぎていないか）、円滑さ",
    "受容・共感",
    "肯定・承認",
    "的確な質問による会話の促進",
    "要約",
    "問題の明確化",
    "この相談での目標の明確化",
    "次の行動につながる提案",
    "勇気づけ・希望の喚起",
]

SYS_PROMPT = """
# タスク説明
あなたには、以下が与えられます。
- カウンセラーとクライアントのSNSカウンセリング対話ログ
- クライアントによる20項目の評価タイトル
あなたの目的は、対話ログに基づいて「クライアントの実際の採点（0〜5）に最も近い推定」を行い、
各項目について推定理由を作成したうえでスコアを出力することです。

# 手順（厳守）
各項目ごとに必ず以下の順で出力する：
1) 日本語で簡潔な感想（1〜3文）：丁寧・具体・行動と感情に言及し、対話で観測できる根拠に寄せる
2) その項目のスコア（0〜5の整数）を決めて出力

## スコア基準（目安）
- 0 = 非常に低い
- 3 = 中立
- 5 = 非常に高い

# 出力フォーマット（厳守）
各項目の出力は **必ず次の2行** とする：
感想: <テキスト>
スコア: <タイトル>=<整数>

## 出力上の注意
- 評価は「助言的」で「非断定」。
- スコア根拠は感想に自然に反映する（説明しすぎない）。
- 過度な自己開示は避ける。
"""

seed = 32


@dataclass
class ConfigInfer:
    # model
    base_model_id: str = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
    adapter_dir: str = f"./outputs/llama3_swallow_8b_qlora_EX_All_V3.0/evalgen-s{seed}"
    use_4bit: bool = True

    # data的路径
    input_json: str = "./datasets/test_data_Part.json"
    out_json: str = f"./gen_results/predictions-s{seed}.json"

    # 参数
    max_new_tokens: int = 192
    temperature: float = 0.2
    top_p: float = 0.95
    do_sample: bool = False
    repetition_penalty: float = 1.05


CFG = ConfigInfer()


_COMMENT_RE = re.compile(r"^\s*感想\s*:\s*(.+)", flags=re.MULTILINE)


def _score_regex_for(title: str) -> re.Pattern:
    # 标题中可能包含全角符号
    esc = re.escape(title)
    return re.compile(rf"^\s*スコア\s*:\s*{esc}\s*=\s*([1-5])\b", flags=re.MULTILINE)


def parse_generation(text: str, title: str) -> Tuple[str, Optional[int]]:
    comment = ""
    m = _COMMENT_RE.search(text)
    if m:
        comment = m.group(1).strip()
    m2 = _score_regex_for(title).search(text)
    score = int(m2.group(1)) if m2 else None
    if score is None:
        m3 = re.search(r"\b([1-5])\b", text)
        score = int(m3.group(1)) if m3 else None
    if not comment:
        comment = "\n".join(
            [ln.strip() for ln in text.strip().splitlines()[:2] if ln.strip()]
        )
    return comment, score


def extract_references(item: Dict[str, Any]) -> Dict[int, Optional[int]]:
    refs: Dict[int, Optional[int]] = {}
    for i in range(1, 21):
        key = f"output_{i}"
        val = item.get(key)
        if not isinstance(val, str):
            refs[i] = None
            continue
        m = re.search(r"=([0-5])\b", val)
        refs[i] = int(m.group(1)) if m else None
    return refs


def build_user_prompt(title: str, dialogue: str) -> str:
    return (
        f"評価項目タイトル: {title}\n"
        f"対話履歴:\n{dialogue}\n\n"
        "上記の内容に基づき、理由とスコアを出力してください。"
    )


def load_tokenizer_and_model(cfg: ConfigInfer):
    quant_cfg = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if cfg.use_4bit
        else None
    )

    # tokenizer: 若适配器存在则优先
    tok_src = (
        cfg.adapter_dir
        if os.path.exists(os.path.join(cfg.adapter_dir, "tokenizer_config.json"))
        else cfg.base_model_id
    )
    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    if tok.pad_token is None and hasattr(tok, "eos_token"):
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # model: AutoPeft → fallback で base+adapter
    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            cfg.adapter_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quant_cfg,
        )
    except Exception:
        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16 if not cfg.use_4bit else None,
            quantization_config=quant_cfg,
        )
        model = PeftModel.from_pretrained(base, cfg.adapter_dir)
    model.eval()
    try:
        model.config.use_cache = True
        model.config.attn_implementation = getattr(
            model.config, "attn_implementation", "sdpa"
        )
    except Exception:
        pass
    return tok, model


def generate_for_title(tok, model, title: str, dialogue: str, cfg: ConfigInfer) -> str:
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": build_user_prompt(title, dialogue)},
    ]
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=cfg.do_sample,
            repetition_penalty=cfg.repetition_penalty,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    # 仅从生成的文本中提取相应
    full = tok.decode(out[0], skip_special_tokens=True)
    cut = full.split("assistant\n", 1)
    return (cut[-1] if len(cut) > 1 else full).strip()


def build_item_obj(
    i: int, title: str, comment: str, score: Optional[int], reference: Optional[int]
) -> Dict[str, Any]:
    return {
        f"evaluation_items_{i}": {
            "label": title,
            "comment": comment,
            "output_score": int(score) if isinstance(score, int) else None,
            "reference": int(reference) if isinstance(reference, int) else None,
        }
    }


def main():
    cfg = CFG
    tok, model = load_tokenizer_and_model(cfg)

    with open(cfg.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    results: List[Dict[str, Any]] = []
    for row in data:
        idx = row.get("index", "")
        dialogue = row.get("input", "")
        refs = extract_references(row)
        print(f"Processing index = {idx}")
        out_obj: Dict[str, Any] = {"index": str(idx), "input": dialogue}
        for i, title in enumerate(EVAL_ITEM_TITLES, start=1):
            gen = generate_for_title(
                tok=tok, model=model, title=title, dialogue=dialogue, cfg=cfg
            )
            comment, score = parse_generation(gen, title)
            if not isinstance(score, int) or not (1 <= score <= 5):
                score = 3  # 未能成功生成则中立判断
            out_obj.update(build_item_obj(i, title, comment, score, refs.get(i)))
        results.append(out_obj)
        print(f"[OK] predicted index = {idx}")

    with open(cfg.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved -> {cfg.out_json} ({len(results)} rows")


if __name__ == "__main__":
    main()
