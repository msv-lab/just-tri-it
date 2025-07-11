import os
import sys


def print_hr():
    width = os.get_terminal_size().columns 
    print('-' * width, file=sys.stderr)


def panic(msg):
    print(msg, file=sys.stderr)
    exit(1)
