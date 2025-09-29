# judge.py
# Simple postcondition that checks whether solution(raw_input) matches expected output.
# This judge uses example mapping only. Replace with a full postcondition for production.

_expected = {
    "6\n3\n3 2 7\n2\n3 3\n5\n5 5 5 5 5\n6\n7 9 3 17 9 13\n3\n6 3 2\n5\n9 4 6 8 3": "27 41 12 \n1 1 \n-1\n1989 1547 4641 819 1547 1071 \n-1\n8 18 12 9 24"
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
