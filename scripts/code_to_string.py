import json
import sys
from pathlib import Path


def python_to_json_string(py_path: str, output_path: str | None = None):
    """
    Convert a Python script into a JSON-safe string value.

    Args:
        py_path (str): Path to the .py file.
        output_path (str | None): Optional output .json file path.
    """
    # 读取源代码
    code = Path(py_path).read_text(encoding="utf-8")

    # 转为 JSON 安全字符串
    json_safe_string = json.dumps(code)

    if output_path:
        Path(output_path).write_text(json_safe_string, encoding="utf-8")
        print(f"✅ Saved JSON-safe string to {output_path}")
    else:
        print(json_safe_string)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python python_to_json_string.py <input.py> [output.json]")
        sys.exit(1)
    py_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    python_to_json_string(py_path, output_path)
