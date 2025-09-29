# judge.py
# Simple postcondition that checks whether solution(raw_input) matches expected output.
# This judge uses example mapping only. Replace with a full postcondition for production.

_expected = {
    "7\n1\n14\n24\n15\n27\n11\n19": "1\n1\n5\n0 -1 0 0 1\n6\n0 0 0 -1 0 1\n5\n-1 0 0 0 1\n6\n-1 0 -1 0 0 1\n5\n-1 0 -1 0 1\n5\n-1 0 1 0 1"
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
