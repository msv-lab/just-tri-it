# judge.py
# Simple postcondition that checks whether solution(raw_input) matches expected output.
# This judge uses example mapping only. Replace with a full postcondition for production.

_expected = {
    "8\n3 4\n4 5\n7 0\n1 1000000000000\n8 14\n112 777\n5 12\n5 2": "Yes\n3 1 2\nNo\nYes\n1 2 3 4 5 6 7\nNo\nYes\n8 2 3 4 5 6 1 7\nNo\nYes\n5 4 3 1 2\nYes\n2 1 3 4 5"
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
