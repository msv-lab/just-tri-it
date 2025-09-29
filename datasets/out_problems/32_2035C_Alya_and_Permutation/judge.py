# judge.py
# Simple postcondition that checks whether solution(raw_input) matches expected output.
# This judge uses example mapping only. Replace with a full postcondition for production.

_expected = {
    "6\n5\n6\n7\n8\n9\n10": "5\n2 1 3 4 5 \n7\n1 2 4 6 5 3 \n7\n2 4 5 1 3 6 7 \n15\n2 4 5 1 3 6 7 8 \n9\n2 4 5 6 7 1 3 8 9 \n15\n1 2 3 4 5 6 8 10 9 7"
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
