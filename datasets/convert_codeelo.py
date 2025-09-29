#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_codeelo.py

Convert codeelo_inexact.json -> test_codeelo_inexact.json
and create per-problem folders containing:
  - requirements.txt
  - inputs.json
  - correct.py
  - incorrect.py
  - judge.py

Each generated test item uses the signature:
    "def solution(raw_input: str) -> str"

Examples are converted into TestFunction assertions.
"""
import json
import re
from pathlib import Path
import argparse

def safe_id(idx, problem):
    """
    Produce a safe id like "1_1971B_Different_String".
    """
    pid = problem.get("problem_id") or problem.get("id") or f"p{idx+1}"
    title = problem.get("title") or ""
    # keep only alphanumeric, underscore and hyphen (hyphens will be replaced by underscore)
    title_clean = re.sub(r'\W+', "_", title).strip("_")
    return f"{idx+1}_{pid}_{title_clean}" if title_clean else f"{idx+1}_{pid}"

def make_test_code(raw_inp, raw_out, test_index=1):
    """
    Return a small pytest-style test function string that calls solution(raw_input).
    """
    return f"def test_{test_index}():\n    assert solution(r'''{raw_inp}''') == r'''{raw_out}'''\n"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def normalize_newlines(s):
    """
    Normalize CRLF to LF and strip trailing newlines.
    Accepts None and returns empty string in that case.
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return s.replace("\r\n", "\n").rstrip("\n")

def main(infile="codeelo_inexact.json", outjson="test_codeelo_inexact.json", outdir="out_problems"):
    infile = Path(infile)
    outjson = Path(outjson)
    outdir = Path(outdir)
    ensure_dir(outdir)

    data = json.loads(infile.read_text(encoding="utf-8"))
    results = []

    for idx, prob in enumerate(data):
        pid = safe_id(idx, prob)
        prob_dir = outdir / pid
        ensure_dir(prob_dir)

        # Use fallback defaults to avoid None issues
        description = prob.get("description") or ""
        input_desc = prob.get("input") or ""
        output_desc = prob.get("output") or ""
        note = prob.get("note") or ""
        rating = prob.get("rating") or ""
        tags = prob.get("tags") or []

        requirements_txt = (
            f"Title: {prob.get('title') or ''}\n"
            f"Problem id: {prob.get('problem_id') or ''}\n"
            f"Rating: {rating}\n"
            f"Tags: {', '.join(tags)}\n\n"
            "Description:\n"
            f"{normalize_newlines(description)}\n\n"
            "Input specification:\n"
            f"{normalize_newlines(input_desc)}\n\n"
            "Output specification:\n"
            f"{normalize_newlines(output_desc)}\n\n"
            "Note:\n"
            f"{normalize_newlines(note)}\n"
        )
        (prob_dir / "requirements.txt").write_text(requirements_txt, encoding="utf-8")

        # Extract examples -> inputs and expected outputs
        examples = prob.get("examples") or []
        sample_pairs = []
        for ex in examples:
            if not isinstance(ex, dict):
                continue
            inp = ex.get("input") or ""
            out = ex.get("output") or ""
            inp = normalize_newlines(inp)
            out = normalize_newlines(out)
            sample_pairs.append((inp, out))

        if not sample_pairs:
            # create an empty placeholder if there are no examples
            sample_pairs.append(("", ""))

        # inputs.json: list of raw input strings (each is used as the single argument to solution)
        inputs_list = [p[0] for p in sample_pairs]
        (prob_dir / "inputs.json").write_text(json.dumps(inputs_list, ensure_ascii=False, indent=2), encoding="utf-8")

        # judge.py: simple postcondition that maps example inputs to example outputs
        mapping = {p[0]: p[1] for p in sample_pairs}
        judge_code = f"""# judge.py
# Simple postcondition that checks whether solution(raw_input) matches expected output.
# This judge uses example mapping only. Replace with a full postcondition for production.

_expected = {json.dumps(mapping, ensure_ascii=False, indent=4)}

def postcondition(raw_input: str, output: str) -> bool:
    # Normalize newlines and whitespace for comparison
    def norm(s: str) -> str:
        return s.replace('\\r\\n', '\\n').strip()
    expected = _expected.get(norm(raw_input), None)
    if expected is None:
        # No ground truth for this input in example mapping -> conservative: reject
        return False
    return norm(output) == norm(expected)
"""
        (prob_dir / "judge.py").write_text(judge_code, encoding="utf-8")

        # correct.py: placeholder "correct" implementation using example mapping
        correct_code = f"""# correct.py
# Placeholder 'correct' implementation that returns example outputs for known example inputs.
# Replace with a real accepted CF solution for broader testing.

def solution(raw_input: str) -> str:
    # Normalize newlines
    inp = raw_input.replace('\\r\\n', '\\n').strip()
    mapping = {json.dumps(mapping, ensure_ascii=False, indent=4)}
    out = mapping.get(inp, None)
    if out is not None:
        return out
    # fallback: empty string (naive)
    return ''
"""
        (prob_dir / "correct.py").write_text(correct_code, encoding="utf-8")

        # incorrect.py: intentionally incorrect implementation for sanity check rejection
        incorrect_code = """# incorrect.py
# Intentionally incorrect implementation for sanity-check rejection.
def solution(raw_input: str) -> str:
    return ''  # always wrong
"""
        (prob_dir / "incorrect.py").write_text(incorrect_code, encoding="utf-8")

        # Build test item for test_codeelo_inexact.json
        signature = "def solution(raw_input: str) -> str"
        req_obj = {
            "id": pid,
            "requirements": {
                "signature": signature,
                "description": requirements_txt
            },
            "tests": [],
            "metadata": {}
        }

        # Each example produces one TestFunction assertion
        for ti, (inp, outp) in enumerate(sample_pairs, start=1):
            test_code = make_test_code(inp, outp, test_index=ti)
            req_obj["tests"].append({
                "type": "TestFunction",
                "code": test_code
            })

        results.append(req_obj)

    # Write aggregated test JSON
    outjson.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Finished: {outjson} and per-problem folders under {outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert codeelo_inexact.json into test items and per-problem scaffolding.")
    parser.add_argument("--in", dest="infile", default="codeelo_inexact.json", help="Input JSON file (default: codeelo_inexact.json)")
    parser.add_argument("--out", dest="outjson", default="test_codeelo_inexact.json", help="Output test JSON file (default: test_codeelo_inexact.json)")
    parser.add_argument("--outdir", dest="outdir", default="out_problems", help="Output directory for per-problem folders (default: out_problems)")
    args = parser.parse_args()
    main(infile=args.infile, outjson=args.outjson, outdir=args.outdir)
