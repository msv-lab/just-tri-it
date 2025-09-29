# judge.py
# Simple postcondition that checks whether solution(raw_input) matches expected output.
# This judge uses example mapping only. Replace with a full postcondition for production.

_expected = {
    "4\n10 10 1\n0 0 3\n-5 -8 8\n4 -5 3": "10 10\n-1 -1\n5 -1\n-4 2\n-6 -7\n-5 -7\n-4 -7\n-4 -8\n-4 -9\n-5 -9\n-6 -9\n-6 -8\n1000 -1000\n-996 995\n8 -10"
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
