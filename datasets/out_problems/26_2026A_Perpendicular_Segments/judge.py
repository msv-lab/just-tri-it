# judge.py
# Simple postcondition that checks whether solution(raw_input) matches expected output.
# This judge uses example mapping only. Replace with a full postcondition for production.

_expected = {
    "4\n1 1 1\n3 4 1\n4 3 3\n3 4 4": "0 0 1 0\n0 0 0 1\n2 4 2 2\n0 1 1 1\n0 0 1 3\n1 2 4 1\n0 1 3 4\n0 3 3 0"
}

def postcondition(raw_input: str, output: str) -> bool:
    # Normalize newlines and whitespace for comparison
    def norm(s: str) -> str:
        return s.replace('\r\n', '\n').strip()
    expected = _expected.get(norm(raw_input), None)
    if expected is None:
        # No ground truth for this input in example mapping -> conservative: reject
        return False
    return norm(output) == norm(expected)
