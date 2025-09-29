# judge.py
# Simple postcondition that checks whether solution(raw_input) matches expected output.
# This judge uses example mapping only. Replace with a full postcondition for production.

_expected = {
    "9\n8\n-1 -1 -1 2 -1 -1 1 -1\n4\n-1 -1 -1 -1\n6\n3 -1 -1 -1 9 -1\n4\n-1 5 -1 6\n4\n2 -1 -1 3\n4\n1 2 3 4\n2\n4 2\n5\n-1 3 -1 3 6\n13\n-1 -1 3 -1 -1 -1 -1 7 -1 -1 3 -1 -1": "4 9 4 2 4 2 1 2\n7 3 6 13\n3 1 2 4 9 18\n-1\n-1\n-1\n4 2\n6 3 1 3 6\n3 1 3 1 3 7 3 7 3 1 3 1 3"
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
