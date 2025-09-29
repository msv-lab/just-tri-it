# judge.py
# Simple postcondition that checks whether solution(raw_input) matches expected output.
# This judge uses example mapping only. Replace with a full postcondition for production.

_expected = {
    "8\ncodeforces\naaaaa\nxxxxy\nco\nd\nnutdealer\nmwistht\nhhhhhhhhhh": "YES\nforcodesec\nNO\nYES\nxxyxx\nYES\noc\nNO\nYES\nundertale\nYES\nthtsiwm\nNO"
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
