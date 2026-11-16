#!/usr/bin/env python3
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["unique_id"]] = row
    return rows


def pct(num: int, den: int) -> str:
    return f"{(100.0 * num / den):.2f}%" if den else "0.00%"


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def response_words(row: dict) -> int:
    return len(str(row.get("response") or "").split())


def short_problem(prompt: str, limit: int = 120) -> str:
    text = prompt.replace("Problem:\n", "", 1).split("\n\nSolution:", 1)[0]
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def add_group(groups: dict, key: str, raw: dict, lora: dict) -> None:
    g = groups[key]
    g["total"] += 1
    g["raw_correct"] += int(bool(raw["correct"]))
    g["lora_correct"] += int(bool(lora["correct"]))
    g["raw_null"] += int(raw.get("pred") is None)
    g["lora_null"] += int(lora.get("pred") is None)
    g["fixed"] += int((not raw["correct"]) and bool(lora["correct"]))
    g["regressed"] += int(bool(raw["correct"]) and (not lora["correct"]))


def table_for_groups(groups: dict[str, dict], key_name: str) -> list[str]:
    lines = [
        f"| {key_name} | total | raw | LoRA | delta | fixed | regressed | raw null | LoRA null |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, g in sorted(groups.items(), key=lambda kv: str(kv[0])):
        total = g["total"]
        raw_acc = 100.0 * g["raw_correct"] / total
        lora_acc = 100.0 * g["lora_correct"] / total
        lines.append(
            "| "
            + " | ".join(
                [
                    str(key),
                    str(total),
                    f"{g['raw_correct']}/{total} = {raw_acc:.2f}%",
                    f"{g['lora_correct']}/{total} = {lora_acc:.2f}%",
                    f"{lora_acc - raw_acc:+.2f} pp",
                    str(g["fixed"]),
                    str(g["regressed"]),
                    f"{g['raw_null']} ({pct(g['raw_null'], total)})",
                    f"{g['lora_null']} ({pct(g['lora_null'], total)})",
                ]
            )
            + " |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--example-limit", type=int, default=5)
    args = parser.parse_args()

    raw_rows = load_jsonl(args.raw)
    lora_rows = load_jsonl(args.lora)
    ids = sorted(set(raw_rows) & set(lora_rows))
    missing_raw = sorted(set(lora_rows) - set(raw_rows))
    missing_lora = sorted(set(raw_rows) - set(lora_rows))

    subject_groups = defaultdict(lambda: defaultdict(int))
    level_groups = defaultdict(lambda: defaultdict(int))

    fixed = []
    regressed = []
    both_correct = []
    both_wrong = []
    raw_word_counts = []
    lora_word_counts = []

    raw_correct = 0
    lora_correct = 0
    raw_null = 0
    lora_null = 0

    for unique_id in ids:
        raw = raw_rows[unique_id]
        lora = lora_rows[unique_id]
        raw_ok = bool(raw["correct"])
        lora_ok = bool(lora["correct"])
        raw_correct += int(raw_ok)
        lora_correct += int(lora_ok)
        raw_null += int(raw.get("pred") is None)
        lora_null += int(lora.get("pred") is None)
        raw_word_counts.append(response_words(raw))
        lora_word_counts.append(response_words(lora))

        add_group(subject_groups, raw.get("subject") or "unknown", raw, lora)
        add_group(level_groups, f"level {raw.get('level')}", raw, lora)

        item = {
            "unique_id": unique_id,
            "subject": raw.get("subject"),
            "level": raw.get("level"),
            "label": raw.get("label"),
            "raw_pred": raw.get("pred"),
            "lora_pred": lora.get("pred"),
            "problem": short_problem(raw.get("prompt", "")),
            "raw_words": response_words(raw),
            "lora_words": response_words(lora),
        }
        if raw_ok and lora_ok:
            both_correct.append(item)
        elif (not raw_ok) and lora_ok:
            fixed.append(item)
        elif raw_ok and (not lora_ok):
            regressed.append(item)
        else:
            both_wrong.append(item)

    total = len(ids)
    summary = {
        "total": total,
        "raw_correct": raw_correct,
        "raw_accuracy": raw_correct / total if total else 0.0,
        "lora_correct": lora_correct,
        "lora_accuracy": lora_correct / total if total else 0.0,
        "absolute_gain": (lora_correct - raw_correct) / total if total else 0.0,
        "fixed": len(fixed),
        "regressed": len(regressed),
        "both_correct": len(both_correct),
        "both_wrong": len(both_wrong),
        "raw_null": raw_null,
        "lora_null": lora_null,
        "raw_response_words_mean": mean(raw_word_counts),
        "lora_response_words_mean": mean(lora_word_counts),
        "raw_response_words_median": median(raw_word_counts),
        "lora_response_words_median": median(lora_word_counts),
        "missing_raw": len(missing_raw),
        "missing_lora": len(missing_lora),
    }

    output = {
        "summary": summary,
        "by_subject": {k: dict(v) for k, v in subject_groups.items()},
        "by_level": {k: dict(v) for k, v in level_groups.items()},
        "fixed_examples": fixed[: args.example_limit],
        "regressed_examples": regressed[: args.example_limit],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Qwen2.5-7B Base Plain LoRA 错误分析",
        "",
        "## 总览",
        "",
        f"- 样本数: {total}",
        f"- raw base: {raw_correct}/{total} = {pct(raw_correct, total)}",
        f"- LoRA iter_0000499: {lora_correct}/{total} = {pct(lora_correct, total)}",
        f"- 绝对提升: {100.0 * summary['absolute_gain']:.2f} pp, 多做对 {lora_correct - raw_correct} 题",
        f"- raw wrong -> LoRA correct: {len(fixed)}",
        f"- raw correct -> LoRA wrong: {len(regressed)}",
        f"- both correct: {len(both_correct)}",
        f"- both wrong: {len(both_wrong)}",
        f"- raw pred=null: {raw_null}/{total} = {pct(raw_null, total)}",
        f"- LoRA pred=null: {lora_null}/{total} = {pct(lora_null, total)}",
        f"- raw response words mean/median: {mean(raw_word_counts):.1f}/{median(raw_word_counts):.1f}",
        f"- LoRA response words mean/median: {mean(lora_word_counts):.1f}/{median(lora_word_counts):.1f}",
        "",
        "## 按 Subject",
        "",
        *table_for_groups(subject_groups, "subject"),
        "",
        "## 按 Level",
        "",
        *table_for_groups(level_groups, "level"),
        "",
        "## LoRA 修复样例",
        "",
    ]
    for item in fixed[: args.example_limit]:
        lines.extend(
            [
                f"- `{item['unique_id']}` {item['subject']} level {item['level']}: label `{item['label']}`, raw `{item['raw_pred']}`, LoRA `{item['lora_pred']}`",
                f"  - {item['problem']}",
            ]
        )

    lines.extend(["", "## LoRA 退化样例", ""])
    for item in regressed[: args.example_limit]:
        lines.extend(
            [
                f"- `{item['unique_id']}` {item['subject']} level {item['level']}: label `{item['label']}`, raw `{item['raw_pred']}`, LoRA `{item['lora_pred']}`",
                f"  - {item['problem']}",
            ]
        )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
