#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq


BOXED_RE = re.compile(r"\\boxed\{")


def extract_boxed(solution: str) -> str | None:
    match = None
    for match in BOXED_RE.finditer(solution):
        pass
    if match is None:
        return None

    start = match.end()
    depth = 1
    i = start
    while i < len(solution):
        ch = solution[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return solution[start:i].strip()
        i += 1
    return None


def parse_level(value: str) -> int | None:
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else None


def make_prompt(problem: str) -> str:
    return (
        "Problem:\n"
        f"{problem.strip()}\n\n"
        "Solution:\n"
    )


def make_completion(solution: str, answer: str) -> str:
    return f"{solution.strip()}\n\nAnswer: \\boxed{{{answer}}}"


def load_rows(path: Path, source: str, include_completion: bool) -> list[dict]:
    rows = pq.read_table(path).to_pylist()
    out = []
    for idx, row in enumerate(rows):
        level = parse_level(row.get("level", ""))
        subject = str(row.get("type", "")).strip()
        solution = str(row.get("solution", "")).strip()
        answer = extract_boxed(solution)
        if level not in {3, 4}:
            continue
        if subject.lower() == "geometry":
            continue
        if not answer:
            continue

        problem = str(row.get("problem", "")).strip()
        item = {
            "prompt": make_prompt(problem),
            "label": answer,
            "source": source,
            "subject": subject,
            "level": level,
            "unique_id": f"{source}_{idx}",
        }
        if include_completion:
            item["metadata"] = {"completion": make_completion(solution, answer)}
        out.append(item)
    return out


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-input",
        type=Path,
        default=Path("/root/datasets/hendrycks-MATH-benchmark/data/train-00000-of-00001.parquet"),
    )
    parser.add_argument(
        "--eval-input",
        type=Path,
        default=Path("/root/datasets/hendrycks-MATH-benchmark/data/test-00000-of-00001.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/root/own/data"))
    parser.add_argument("--train-seed", type=int, default=5151)
    parser.add_argument("--eval-seed", type=int, default=6060)
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[1000, 2000])
    parser.add_argument("--eval-size", type=int, default=600)
    args = parser.parse_args()

    train_rows = load_rows(args.train_input, "hendrycks_math_train_plain", include_completion=True)
    rng = random.Random(args.train_seed)
    rng.shuffle(train_rows)
    print(f"eligible_train_rows={len(train_rows)}")
    if len(train_rows) < max(args.train_sizes):
        raise SystemExit(f"Need {max(args.train_sizes)} train rows, found {len(train_rows)}.")
    for size in args.train_sizes:
        out = args.output_dir / f"math_plain_sft_l3_4_core_train{size}_seed{args.train_seed}.jsonl"
        write_jsonl(train_rows[:size], out)
        print(f"wrote train {size} -> {out}")

    eval_rows = load_rows(args.eval_input, "hendrycks_math_test_plain", include_completion=False)
    rng = random.Random(args.eval_seed)
    rng.shuffle(eval_rows)
    print(f"eligible_eval_rows={len(eval_rows)}")
    if len(eval_rows) < args.eval_size:
        raise SystemExit(f"Need {args.eval_size} eval rows, found {len(eval_rows)}.")
    out = args.output_dir / f"math_plain_eval_l3_4_core_eval{args.eval_size}_seed{args.eval_seed}.jsonl"
    write_jsonl(eval_rows[: args.eval_size], out)
    print(f"wrote eval {args.eval_size} -> {out}")


if __name__ == "__main__":
    main()
