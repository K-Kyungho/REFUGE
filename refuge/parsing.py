"""Parsing of LLM outputs: feature plans, judge selection, generated code."""

import json
import re
from typing import Any, Dict, List, Tuple

_FEATURE_PAT = re.compile(
    r"Feature\s*\d+\s*:\s*"
    r"(?:-+\s*)?"
    r"(?:Name\s*:\s*(?P<name>.+?)\n)?"
    r"(?:.*?Plan\s*:\s*(?P<plan>.+?))"
    r"(?:\n-?\s*Rationale\s*:\s*(?P<rat>.+?))?(?:\n\n|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_feature_plans(text: str, iter_idx: int, src_model: str) -> List[Dict[str, Any]]:
    """Extract up to 3 (Name / Plan / Rationale) blocks from a Step-2 response.

    Falls back to any paragraph mentioning "plan" if the strict format is not matched.
    """
    def item(name, plan, rat=""):
        return {"iter": iter_idx, "src_model": src_model,
                "name_raw": name, "plan": plan, "rationale": rat}

    items = []
    for m in _FEATURE_PAT.finditer(text):
        plan = (m.group("plan") or "").strip()
        if plan:
            items.append(item((m.group("name") or f"feat{iter_idx}_?").strip(),
                              plan, (m.group("rat") or "").strip()))
    if not items:
        for chunk in re.split(r"\n\s*\n", text):
            if "plan" in chunk.lower():
                items.append(item(f"feat{iter_idx}_?", chunk.strip()))
    return items[:3]


def parse_judge(text: str) -> Tuple[List[int], Dict[str, str]]:
    """Parse the judge's JSON; fall back to the first three candidates."""
    selected, reasons = [], {}
    try:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            obj = json.loads(m.group(0))
            selected = obj.get("selected_ids", [])
            reasons = obj.get("reasons", {})
    except Exception:
        print("[WARN] judge JSON parse failed; falling back to first 3.")
        return [1, 2, 3], {}
    if not selected or len(selected) < 3:
        print("[WARN] judge returned insufficient ids; falling back to first 3.")
        selected = [1, 2, 3]
    return selected, reasons


def extract_python(text: str) -> str:
    m = re.search(r"```python\s*(.*?)\s*```", text, re.S)
    return m.group(1) if m else text
