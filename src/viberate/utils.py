import sys
import shutil


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
