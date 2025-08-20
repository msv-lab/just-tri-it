import sys
import shutil
import mistletoe


def print_hr():
    width = shutil.get_terminal_size(fallback=(80, 20)).columns
    print('-' * width, file=sys.stderr)


def print_annotated_hr(message):
    width = shutil.get_terminal_size(fallback=(80, 20)).columns
    msg = f' {message} '
    dash_count = width - len(msg)
    if dash_count < 0:
        print(message)
        return
    left_dashes = dash_count // 2
    right_dashes = dash_count - left_dashes
    line = '-' * left_dashes + msg + '-' * right_dashes
    print(line, file=sys.stderr)


def panic(msg):
    print(msg, file=sys.stderr)
    exit(1)


class DataExtractionFailure(Exception):
    "Raised when failed to parse LLM output"
    pass    
    

def extract_code(content):
    """Extract first markdown code block"""
    parsed = mistletoe.Document(content)
    for child in parsed.children:
        if child.__class__.__name__ == "CodeFence":
            return child.children[0].content
    raise DataExtractionFailure


def extract_answer(s):
    if "<answer>" in s and "</answer>" in s and \
       s.index("<answer>") < s.index("</answer>"):
        return s.split("<answer>", 1)[1].split("</answer>", 1)[0]
    else:
        raise DataExtractionFailure
