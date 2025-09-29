# judge.py
# Simple postcondition that checks whether solution(raw_input) matches expected output.
# This judge uses example mapping only. Replace with a full postcondition for production.

_expected = {
    "10\n6\nNENSNE\n3\nWWW\n6\nNESSWS\n2\nSN\n2\nWE\n4\nSSNN\n4\nWESN\n2\nSS\n4\nEWNN\n4\nWEWE": "RRHRRH\nNO\nHRRHRH\nNO\nNO\nRHRH\nRRHH\nRH\nRRRH\nRRHH"
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
