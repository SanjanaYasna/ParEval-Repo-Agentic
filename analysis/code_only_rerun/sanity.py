import argparse
import html
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

FILE_HEADER_PATTERNS = [
    # [foo.cpp]
    re.compile(r"^\[(?P<name>[^\]\[]+)\]\s*$"),
    # [[ReadMe.md](http://ReadMe.md)]  — markdown link syntax
    re.compile(r"^\[\[(?P<name>[^\]]+)\]\([^)]*\)\]\s*$"),
]

FENCE_START_RE = re.compile(r"^```[A-Za-z0-9_+\-]*\s*$")
FENCE_END_RE   = re.compile(r"^```\s*$")

def _sanitize_rel_path(raw: str) -> Optional[str]:
    raw = raw.strip()
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        return None
    if any(part == ".." for part in p.parts):
        return None
    return str(p)

def parse_llm_response(text: str) -> Dict[str, str]:
    """
    Parse [FileName] / ```lang ... ``` blocks out of an LLM response.

    Returns {relative_path: file_content}.
    """
    lines = text.splitlines()
    out: Dict[str, str] = {}
    i = 0

    while i < len(lines):
        raw_line = lines[i].strip()
        filename: Optional[str] = None

        for pat in FILE_HEADER_PATTERNS:
            m = pat.match(raw_line)
            if m:
                filename = m.group("name").strip()
                break

        if filename is None:
            i += 1
            continue

        i += 1

        # skip blank lines between header and fence
        while i < len(lines) and not lines[i].strip():
            i += 1

        if i >= len(lines) or not FENCE_START_RE.match(lines[i].strip()):
            # no fence found — the header line was probably something else
            continue

        i += 1  # skip opening fence
        buf: List[str] = []

        while i < len(lines) and not FENCE_END_RE.match(lines[i].strip()):
            buf.append(lines[i])
            i += 1

        if i < len(lines):
            i += 1  # skip closing fence

        safe = _sanitize_rel_path(filename)
        if safe is None:
            continue

        content = html.unescape("\n".join(buf)).rstrip("\n") + "\n"
        out[safe] = content
    print("PARSED RESPONSE")
    print(out)
    return out

if __name__ == "__main__":
    with open("/Users/robsonlab/scratch/optimize_hpx_to_legion/AsyncSTM/opus/output_8/COMPILATION/llm_response_iter_001.txt", "r", encoding="utf-8") as f:
        text = f.read()
    extracted = parse_llm_response(text)
    print(extracted)
        