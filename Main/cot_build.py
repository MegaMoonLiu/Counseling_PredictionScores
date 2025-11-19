# path: scripts/cot_build_and_train.py
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

EVAL_COUNT = 20
SRC_DEFAULT = Path("./datasets/All_data_EX_Part_P.json")
OUT_DEFAULT = Path("./datasets/cot_swallow_P.jsonl")
MODEL_DEFAULT = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"

SYS_PROMPT = (
    "あなたは臨床対話を評価するアシスタントです。"
    "各項目で先に妥当な理由を述べ、その後に0-5の整数スコアのみをスコアで明示してください。"
)

USER_TMPL = (
    "以下は心理相談の対話履歴です。\n"
    "入力（対話履歴）:\n{dialog}\n\n"
    "出力タスク:\n"
    "20項目について、各項目の「理由 -> スコア(0-5)」の順で、1行ずつ日本語で記述してください。\n"
    "【出力フォーマット（厳守）】\n"
    "evaluation_items_1: 不安や無力感に丁寧に寄り添い、内容の要約で理解を確かめてくれたため、受け止められている実感があった。加えて、感情の強さや背景（例：一人で頑張りたい理由、眠りの質の変化）に一歩踏み込んだ反映があると、さらに伝わりやすい。\n"
    "output_1: 聴いてもらえた、わかってもらえたと感じた=4\n"
    "evaluation_items_2: 一人でやり抜きたい意向や実家に戻らない選択を尊重しつつ、助けを借りる選択肢も提案していた。提案時に「今はどれが現実的か」「どこまで試したいか」を都度確認すると、主体性がより保たれる。\n"
    "output_2: 尊重されたと感じた=4\n"
    "evaluation_items_3: 配信ライブの活用や公的支援に頼る視点、自己を労わる時間の重要性など、新しい捉え方が得られた。睡眠や不安へのセルフケア、家計の見通しづくりなど具体手法も紹介されると、気づきが行動につながりやすい。\n"
    "output_3: 新しい気づきや体験があった=4\n"
    "evaluation_items_4: 「少し調べてみる」といった小さな一歩につながった。励ましも温かかった。負担を感じやすい表現には「無理のない範囲で」などのクッションを添えると、前向きさが持続する。\n"
    "output_4: 希望や期待を感じられた=4\n"
    "evaluation_items_5: 収入減・将来不安・孤立感といった核心に触れ、家族への共有の可否や生活の見通しも扱われた。加えて、就労支援や具体的な経済支援へのアクセスまで扱えると、取り組みたい話題が一層網羅される。\n"
    "output_5: 取り組みたかったことを扱えた=4\n"
    "evaluation_items_6: 現状整理から価値観の確認、対処案の検討まで一緒に進められた。最後に「今週の一歩」を共同で決め、実行条件や障害への備えを整えると、協働感がより強まる。\n"
    "output_6: 一緒に考えながら取り組めた=4\n"
    "evaluation_items_7: 落ち着いたテンポで安心して話せた。反復共感が続く場面では、感情の言い換えや具体の要約に変化をつけると、さらに心地よい流れになる。\n"
    "output_7: やりとりのリズムがあっていた=4\n"
    "evaluation_items_8: 否定のない雰囲気で居心地が良かった。適度な自己開示で距離が縮まった一方、分量は控えめにしつつ焦点をクライアントに戻す合図があると、安心感が保たれる。\n"
    "output_8: 居心地のよいやりとりだった=4\n"
    "evaluation_items_9: 気持ちの受け止めと現実的な提案のバランスが良かった。最後にセッションの要点と次の一歩を確認できると、全体のまとまりがさらに高まる。\n"
    "output_9: 全体として適切でよかった=4\n"
    "evaluation_items_10: 愚痴を安全に吐き出せ、試せるヒントが得られた。地域資源やオンライン資源の具体名を添えると、価値がいっそう実感しやすい。\n"
    "output_10: 今回の相談は価値があった=4\n"
    "evaluation_items_11: 開始の合意形成と導入がスムーズだった。冒頭で「今日は何を達成できたら良いか」を短く共有すると、方向性がさらに明確になる。\n"
    "output_11: 相談開始の円滑さ=4\n"
    "evaluation_items_12: 感想の確認と労いが丁寧だった。次回までの具体的な一歩、再相談の方法、緊急時の連絡先などを明示し、簡単なまとめで締めると、納得感が増す。\n"
    "output_12: 相談終了のタイミング（不必要に聴きすぎていないか）、円滑さ=3\n"
    "evaluation_items_13: 不安・無力感・孤立感に継続的に寄り添っていた。クライアントの言葉を用いた感情の反映や、身体反応（睡眠・倦怠）の共感まで触れると、より深く伝わる。\n"
    "output_13: 受容・共感=4\n"
    "evaluation_items_14: 節約や一人で踏ん張っている努力をきちんと認めていた。具体的な行動や強みを列挙して承認すると、自己効力感がさらに高まる。\n"
    "output_14: 肯定・承認=4\n"
    "evaluation_items_15: 仕事・収入・生活の見通し・支えの有無など、要点に届く質問があった。優先順位を問う、数値で状態を測る（睡眠の質を0～10で等）質問を加えると、整理が進む。\n"
    "output_15: 的確な質問による会話の促進=4\n"
    "evaluation_items_16: 中盤での要点整理が分かりやすく、理解の一致が取れていた。要約後に「他に大事な点はありますか？」と確認まで行うと、さらに有効になる。\n"
    "output_16: 要約=4\n"
    "evaluation_items_17: 収入減・外出制限・孤立が不安を強めている構図が示された。短期（睡眠・気分の維持）と中期（収支の安定）の課題を区別して言語化すると、見通しがより明瞭になる。\n"
    "output_17: 問題の明確化=4\n"
    "evaluation_items_18: 「どうなったら良いか」の問いかけはあったが、答えづらい状態に合わせた選択肢提示や小目標化には十分至らなかった。共同で「今週は配信ライブを1つ視聴する」等の具体目標を合意できると良い。\n"
    "output_18: この相談での目標の明確化=3\n"
    "evaluation_items_19: 配信ライブの活用、公的・家族支援へのアクセスといった現実的な提案があった。具体窓口（自立相談支援機関、生活福祉資金、自治体の家計相談、ハローワークの職業相談）や睡眠衛生のコツを添えると、実行性が高まる。\n"
    "output_19: 次の行動につながる提案=4\n"
    "evaluation_items_20: 温かい励ましで、行動への意欲がわいた。できている点（節約・情報収集の再開）を明確にフィードバックし、進捗を一緒に振り返る提案があると、希望がより強固になる。\n"
    "output_20: 勇気づけ・希望の喚起=4\n"
)

NAME_SCORE_RE = re.compile(r"^\s*(?P<name>.+?)\s*=\s*(?P<score>[0-5])\s*$")


@dataclass
class BuiltSample:
    messages: List[Dict[str, str]]


def parse_name_score(raw: Any) -> Tuple[str, str]:
    m = NAME_SCORE_RE.match(str(raw))
    if not m:
        raise ValueError(f"无法解析评分（缺少 '=0..5'）: {raw!r}")
    return m.group("name").strip(), m.group("score").strip()


def build_cot_lines(item: Dict[str, Any]) -> str:
    lines: List[str] = []
    for i in range(1, EVAL_COUNT + 1):
        reason_key = f"evaluation_items_{i}"
        output_key = f"output_{i}"
        if reason_key not in item or output_key not in item:
            raise KeyError(f"缺少键: {reason_key} 或 {output_key}")
        name, score = parse_name_score(item[output_key])
        reason = str(item[reason_key]).strip()
        lines.append(f"{name}: {reason} -> {score}")  # CoT: 理由 -> スコア
    return "\n".join(lines)


def build_one(item: Dict[str, Any]) -> BuiltSample:
    dialog = str(item.get("input", "")).strip()
    if not dialog:
        raise ValueError("对话历史为空：字段 'input' 缺失或为空。")
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": USER_TMPL.format(dialog=dialog)},
        {"role": "assistant", "content": build_cot_lines(item)},
    ]
    return BuiltSample(messages=messages)


def build_jsonl(src: Path, out_path: Path, strict: bool = True) -> int:
    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError("源 JSON 必须是数组。")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as fw:
        for idx, item in enumerate(data):
            try:
                sample = build_one(item)
                fw.write(
                    json.dumps({"messages": sample.messages}, ensure_ascii=False) + "\n"
                )
                n += 1
            except Exception as e:
                if strict:
                    raise RuntimeError(f"样本 {idx} 构建失败: {e}") from e
                print(f"[WARN] 跳过样本 {idx}: {e}")
    print(f"[OK] 写出 {n} 条 => {out_path.resolve()}")
    return n


def main():
    ap = argparse.ArgumentParser(
        description="Build CoT JSONL from input & train Swallow-8B with LoRA"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_b = sub.add_parser("build", help="构建 CoT JSONL（messages）")
    p_b.add_argument("--src", type=Path, default=SRC_DEFAULT)
    p_b.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p_b.add_argument("--non-strict", action="store_true")

    args = ap.parse_args()
    build_jsonl(args.src, args.out, strict=not args.non_strict)


if __name__ == "__main__":
    main()
