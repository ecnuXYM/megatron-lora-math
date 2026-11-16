#!/usr/bin/env python3
import argparse
import json
import re
import tempfile
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


BOXED_RE = re.compile(r"\\boxed\{")


def extract_boxed(text: str) -> str | None:
    match = None
    for match in BOXED_RE.finditer(text):
        pass
    if match is None:
        return None
    start = match.end()
    depth = 1
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None


def normalize_answer(x: str | None) -> str:
    if x is None:
        return ""
    x = x.strip()
    x = x.replace("\\left", "").replace("\\right", "")
    x = x.replace(" ", "").replace("\n", "")
    x = x.rstrip(".")
    return x


def is_correct(pred: str | None, label: str) -> bool:
    return normalize_answer(pred) == normalize_answer(label)


def load_jsonl(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_existing_results(path: Path) -> tuple[set[str], int, int]:
    seen = set()
    correct = 0
    total = 0
    if not path.is_file():
        return seen, correct, total

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            unique_id = row.get("unique_id")
            if unique_id is None:
                continue
            seen.add(unique_id)
            correct += int(bool(row.get("correct")))
            total += 1
    return seen, correct, total


def prepare_peft_adapter(adapter_dir: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    weight_path = adapter_dir / "adapter_model.bin"
    config_path = adapter_dir / "adapter_config.json"
    if not weight_path.is_file() or not config_path.is_file():
        return adapter_dir, None

    state = torch.load(weight_path, map_location="cpu")
    if not isinstance(state, dict) or not state:
        return adapter_dir, None

    first_key = next(iter(state))
    if first_key.startswith("base_model.model."):
        return adapter_dir, None

    tmp = tempfile.TemporaryDirectory(prefix="miles_peft_adapter_")
    out_dir = Path(tmp.name)

    converted = {}
    target_modules = set()
    for key, value in state.items():
        new_key = key
        if new_key.startswith("model."):
            new_key = "base_model.model." + new_key
        converted[new_key] = value

        match = re.search(r"self_attn\.([a-z]_proj)\.lora_[AB]", key)
        if match:
            target_modules.add(match.group(1))

    config = json.loads(config_path.read_text())
    if target_modules:
        config["target_modules"] = sorted(target_modules)
    config.setdefault("peft_type", "LORA")
    config.setdefault("task_type", "CAUSAL_LM")

    torch.save(converted, out_dir / "adapter_model.bin")
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return out_dir, tmp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--resume", action="store_true", help="Append missing rows if output already exists.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.adapter is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise SystemExit("peft is required for --adapter. Run: pip install peft") from exc
        adapter_dir, tmp_adapter = prepare_peft_adapter(args.adapter)
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    else:
        tmp_adapter = None

    model.eval()
    rows = load_jsonl(args.data, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        seen_ids, correct, total = load_existing_results(args.output)
        rows_to_run = [row for row in rows if row.get("unique_id") not in seen_ids]
        mode = "a"
        print(f"resume: existing={total}, remaining={len(rows_to_run)}")
    else:
        correct = 0
        total = 0
        rows_to_run = rows
        mode = "w"

    with args.output.open(mode, encoding="utf-8") as f:
        for row in tqdm(rows_to_run):
            prompt = row["prompt"]
            inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            pred = extract_boxed(response)
            ok = is_correct(pred, row["label"])
            correct += int(ok)
            total += 1
            f.write(
                json.dumps(
                    {
                        "unique_id": row.get("unique_id"),
                        "subject": row.get("subject"),
                        "level": row.get("level"),
                        "label": row["label"],
                        "pred": pred,
                        "correct": ok,
                        "prompt": prompt,
                        "response": response,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()

    summary = {
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "data": str(args.data),
        "output": str(args.output),
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    with args.output.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    if tmp_adapter is not None:
        tmp_adapter.cleanup()


if __name__ == "__main__":
    main()
