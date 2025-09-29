# judge.py
# Simple postcondition that checks whether solution(raw_input) matches expected output.
# This judge uses example mapping only. Replace with a full postcondition for production.

_expected = {
    "4\n2\n1 4\n2 3\n3\n3 2\n4 3\n2 1\n5\n5 10\n2 3\n9 6\n4 1\n8 7\n1\n10 20": "2 3 1 4\n2 1 3 2 4 3\n4 1 2 3 5 10 8 7 9 6\n10 20"
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
