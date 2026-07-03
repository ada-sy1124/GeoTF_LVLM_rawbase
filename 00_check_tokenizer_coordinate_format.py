"""
Check how one tokenizer splits coordinate strings.

This script only does three things:
1. load the tokenizer;
2. tokenize the coordinate strings below;
3. save token ids, token strings, and decoded pieces.

Edit the config below, then run:

    python 00_check_tokenizer_coordinate_format.py
"""

import json
import re
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer


# =============================
# Editable config
# =============================
# Run one model at a time. Change MODEL_PATH here when switching tokenizer.
# MODEL_PATH = "llava-hf/llava-1.5-7b-hf"
MODEL_PATH = "llava-hf/llava-1.5-13b-hf"
# MODEL_PATH = "lmms-lab/llava-onevision-qwen2-7b-ov"
# MODEL_PATH = "nyu-visionx/cambrian-8b"
# MODEL_PATH = "YanweiLi/MGM-13B"
# MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
# MODEL_PATH = "Qwen/Qwen3.5-VL-8B-Instruct"

TRUST_REMOTE_CODE = True
OUTPUT_DIR = "tokenizer_coordinate_reports"

COORDINATES = [
    "1.2345, 2.3456",
    "25.7734, 55.4567",
    "25.7734, 155.4567",
    "-25.7734, 155.4567",
    "25.7734, -155.4567",
    "-25.7734, -155.4567",
    "89.9999, 179.9999",
    "-89.9999, -179.9999",
    "0.0000, 0.0000",
    "-6.7890, 55.4678",
    "12.3456, -123.4567",
]


def slugify(text):
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip("/"))
    return slug.strip("_") or "tokenizer"


def inspect_coordinate(tokenizer, coordinate):
    token_ids = tokenizer.encode(coordinate, add_special_tokens=False)
    token_ids = [int(token_id) for token_id in token_ids]
    token_strings = tokenizer.convert_ids_to_tokens(token_ids)
    decoded_pieces = [
        tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for token_id in token_ids
    ]

    return {
        "coordinate": coordinate,
        "token_count": len(token_ids),
        "token_ids": token_ids,
        "token_strings": token_strings,
        "decoded_pieces": decoded_pieces,
        "tokens": [
            {
                "index": index,
                "id": token_id,
                "token": token,
                "decoded": decoded,
            }
            for index, (token_id, token, decoded) in enumerate(
                zip(token_ids, token_strings, decoded_pieces)
            )
        ],
    }


def build_report(tokenizer):
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": "00_check_tokenizer_coordinate_format.py",
        "model_path": MODEL_PATH,
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "add_special_tokens": False,
        "coordinates": [
            inspect_coordinate(tokenizer, coordinate)
            for coordinate in COORDINATES
        ],
    }


def markdown_escape(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown(report):
    lines = [
        "# Coordinate Tokenizer Report",
        "",
        f"- model: `{report['model_path']}`",
        f"- tokenizer: `{report['tokenizer_class']}`",
        f"- vocab_size: `{report['vocab_size']}`",
        "- add_special_tokens: `False`",
        "",
    ]

    for item in report["coordinates"]:
        lines.extend([
            f"## `{item['coordinate']}`",
            "",
            f"- token_count: `{item['token_count']}`",
            "",
            "| idx | id | token | decoded |",
            "| ---: | ---: | --- | --- |",
        ])
        for token in item["tokens"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(token["index"]),
                        str(token["id"]),
                        f"`{markdown_escape(token['token'])}`",
                        f"`{markdown_escape(token['decoded'])}`",
                    ]
                )
                + " |"
            )
        lines.append("")

    return "\n".join(lines)


def save_report(report):
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = slugify(MODEL_PATH)
    json_path = output_dir / f"{name}_coordinate_tokenization.json"
    md_path = output_dir / f"{name}_coordinate_tokenization.md"

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def main():
    print(f"Model path: {MODEL_PATH}")
    print(f"Coordinates: {len(COORDINATES)}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=TRUST_REMOTE_CODE,
        use_fast=False,
    )
    report = build_report(tokenizer)
    json_path, md_path = save_report(report)

    print(f"Saved JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()




# python ./00_check_tokenizer_coordinate_format.py