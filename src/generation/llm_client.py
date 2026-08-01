from __future__ import annotations
import os
import re
import json
import requests
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    response = requests.post(
        f"{BASE_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=420,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]
def parse_llm_json(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in LLM output: {raw_text[:200]!r}")
    candidate = cleaned[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return json.loads(_repair_inner_quotes(candidate))
def _repair_inner_quotes(candidate: str) -> str:
    out = []
    in_string = False
    i = 0
    n = len(candidate)
    while i < n:
        ch = candidate[i]
        if ch == "\\" and i + 1 < n:
            out.append(ch)
            out.append(candidate[i + 1])
            i += 2
            continue
        if ch == '"':
            if not in_string:
                in_string = True
                out.append(ch)
            else:
                j = i + 1
                while j < n and candidate[j] in " \t\n\r":
                    j += 1
                if j < n and candidate[j] in ",:}]":
                    in_string = False
                    out.append(ch)
                else:
                    out.append('\\"')
        else:
            out.append(ch)
        i += 1
    return "".join(out)